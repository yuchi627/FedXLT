import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch import eq, no_grad
from torch.utils.data import DataLoader

from Model.prototree.prototree import ProtoTree
from util.acc_calculate import accuracy


@torch.no_grad()
def eval(tree: ProtoTree,
        test_loader: DataLoader,
        device,
        cls_num,
        sampling_strategy: str = 'distributed',
        ) -> dict:

    tree = tree.to(device)
    # Make sure the model is in evaluation mode
    tree.eval()
    pre_list = [0 for _ in range(cls_num)]
    cor_list = [0 for _ in range(cls_num)]
    lab_list = [0 for _ in range(cls_num)]
    eval_info = {}
    with no_grad():
        acc1_avg = 0
        acc3_avg = 0 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, test_info = tree.forward(images, sampling_strategy)
            acc1, acc3 = accuracy(outputs, labels, topk=(1, 3))
            acc1_avg += acc1
            acc3_avg += acc3
            _, predicts = torch.max(outputs, -1)
            for (pre, cor, lab) in zip(predicts, eq(predicts.cpu(), labels.cpu()), labels):
                pre_list[pre]+=1
                if cor:
                    cor_list[pre]+=1
                lab_list[lab] += 1
        acc1_avg = acc1_avg / float(len(test_loader))
        acc3_avg = acc3_avg / float(len(test_loader))
        # predict_dict = {str(c):num for c, num in enumerate(pre_list)}
        # correct_dict = {str(c):num for c, num in enumerate(cor_list)}
        # label_dict = {str(c):num for c, num in enumerate(lab_list)}
        correct_ratio_dict={}
        # correct_ratio={"Few":0,"Median":0,"Many":0}
        for c, (cor, lab) in enumerate(zip(cor_list, lab_list)):
            if (not lab==0): 
                correct_ratio_dict[str(c)] = float(cor)/float(lab)*100
            else :
                correct_ratio_dict[str(c)] = -1 
    print(" Eval Acc1: ", acc1_avg)
    print(" correct_ratio_dict: ", correct_ratio_dict)
    eval_info["test_accuracy"] = acc1_avg
    eval_info["correct_ratio_dict"] = correct_ratio_dict
    return eval_info





@torch.no_grad()
def eval_ori(tree: ProtoTree,
        test_loader: DataLoader,
        epoch,
        device,
        cls_num,
        sampling_strategy: str = 'distributed',
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    tree = tree.to(device)
    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((tree._num_classes, tree._num_classes), dtype=int)

    # Make sure the model is in evaluation mode
    tree.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)

    class_acc = {}
    for c in range(cls_num):
        class_acc[c] = 0.
    one = torch.tensor(1).to(device)
    zero = torch.tensor(0).to(device)
    neg = torch.tensor(-1).to(device)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        out, test_info, _ = tree.forward(xs, sampling_strategy)
        ys_pred = torch.argmax(out, dim=1)

        # Update the confusion matrix
        cm_batch = np.zeros((tree._num_classes, tree._num_classes), dtype=int)
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1
        acc = acc_from_cm(cm_batch)

        y_num = torch.where(y_pred == ys, ys, neg)
        uni_y_list, count_list = torch.unique(ys, return_counts=True)
        for uni_y, count in zip(uni_y_list, count_list):
            y_num_uni = torch.sum(torch.where(y_num == uni_y.expand(y_num.shape), one, zero))
            class_acc[int(uni_y)] += float(torch.true_divide(y_num_uni,count))



        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc:.3f}'
        )

        # keep list of leaf indices where test sample ends up when deterministic routing is used.
        if sampling_strategy != 'distributed':
            info['out_leaf_ix'] += test_info['out_leaf_ix']
        del out
        del ys_pred
        del test_info
    for c in range(cls_num):
        class_acc[c] = class_acc[c]/float(i+1)
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    info['class_accuracy'] = class_acc
    print("\nEpoch %s - Test accuracy with %s routing: "%(epoch, sampling_strategy)+str(info['test_accuracy']))
    return info

@torch.no_grad()
def eval_fidelity(tree: ProtoTree,
        test_loader: DataLoader,
        device,
        progress_prefix: str = 'Fidelity'
        ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()

    # Make sure the model is in evaluation mode
    tree.eval()
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix,
                        ncols=0)

    distr_samplemax_fidelity = 0
    distr_greedy_fidelity = 0
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data, with 3 types of routing
        out_distr, _ = tree.forward(xs, 'distributed')
        ys_pred_distr = torch.argmax(out_distr, dim=1)

        out_samplemax, _ = tree.forward(xs, 'sample_max')
        ys_pred_samplemax = torch.argmax(out_samplemax, dim=1)

        out_greedy, _ = tree.forward(xs, 'greedy')
        ys_pred_greedy = torch.argmax(out_greedy, dim=1)
        
        # Calculate fidelity
        distr_samplemax_fidelity += torch.sum(torch.eq(ys_pred_samplemax, ys_pred_distr)).item()
        distr_greedy_fidelity += torch.sum(torch.eq(ys_pred_greedy, ys_pred_distr)).item()
        # Update the progress bar
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}]'
        )
        del out_distr
        del out_samplemax
        del out_greedy

    distr_samplemax_fidelity = distr_samplemax_fidelity/float(len(test_loader.dataset))
    distr_greedy_fidelity = distr_greedy_fidelity/float(len(test_loader.dataset))
    info['distr_samplemax_fidelity'] = distr_samplemax_fidelity
    info['distr_greedy_fidelity'] = distr_greedy_fidelity
    print("Fidelity between standard distributed routing and sample_max routing: "+str(distr_samplemax_fidelity))
    print("Fidelity between standard distributed routing and greedy routing: "+str(distr_greedy_fidelity))
    return info

@torch.no_grad()
def eval_ensemble(trees: list, test_loader: DataLoader, device, args: argparse.Namespace, sampling_strategy: str = 'distributed', progress_prefix: str = 'Eval Ensemble'):
    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((trees[0]._num_classes, trees[0]._num_classes), dtype=int)    

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix,
                        ncols=0)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        outs = []
        for tree in trees:
            # Make sure the model is in evaluation mode
            tree.eval()
            tree = tree.to(device)
            # Use the model to classify this batch of input data
            out, _ = tree.forward(xs, sampling_strategy)
            outs.append(out)
            del out
        stacked = torch.stack(outs, dim=0)
        ys_pred = torch.argmax(torch.mean(stacked, dim=0), dim=1)
        
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}]'
        )
        del outs
            
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    print("Ensemble accuracy with %s routing: %s"%(sampling_strategy, str(info['test_accuracy'])))
    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
