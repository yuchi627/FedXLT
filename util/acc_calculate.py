from ipaddress import v4_int_to_packed
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm



@torch.no_grad()
def eval(net,
        add_on,
        cls,
        test_loader: DataLoader,
        epoch,
        device,
        cls_num,
        args,
        sampling_strategy: str = 'distributed',
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    net = net.to(device)
    add_on = add_on.to(device)
    cls = cls.to(device)

    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((args.num_classes, args.num_classes), dtype=int)

    # Make sure the model is in evaluation mode
    net.eval()
    add_on.eval()
    cls.eval()

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
        # out, test_info, _ = tree.forward(xs, sampling_strategy)
        features = net(xs)
        features = add_on(features)
        if args.client_classifier == "tree":
            out, test_info, _  = cls(features)
        else:
            out = cls(features)
        ys_pred = torch.argmax(out, dim=1)

        # Update the confusion matrix
        cm_batch = np.zeros((args.num_classes, args.num_classes), dtype=int)
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1
        acc = acc_from_cm(cm_batch)

        # keep list of leaf indices where test sample ends up when deterministic routing is used.
        if args.client_classifier == "tree":
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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

