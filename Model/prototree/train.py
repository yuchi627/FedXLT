from tqdm.auto import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from Model.prototree.prototree import ProtoTree
from util.methods import *
from util.log import Log

def train_epoch(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                args: argparse,
                num_class_list: list,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:
    
    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    nor_total_loss = 0.
    hcm_total_loss = 0.
    smoothing_total_loss = 0.
    mixup_total_loss = 0.
    class_acc = {}
    for c in range(len(num_class_list)):
        class_acc[c] = 0.
    one = torch.tensor(1).to(device)
    zero = torch.tensor(0).to(device)
    neg = torch.tensor(-1).to(device)

    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = float(len(train_loader))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+' %s'%epoch,
                    position=0, 
                    leave=True,
                    # ncols=0
                    )
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        # Perform a forward pass through the network
        ys_pred, info, ys_logits = tree.forward(xs)

        # Learn prototypes and network with gradient descent. 
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        nor_loss = 0.
        hcm_loss = 0.
        mixup_loss = 0.
        smoothing_loss = 0.
        if tree._log_probabilities:
            nor_loss = F.nll_loss(ys_pred, ys)
        else:       
            nor_loss = F.nll_loss(torch.log(ys_pred), ys)
        loss = nor_loss +  hcm_loss + mixup_loss + smoothing_loss
        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()
        
        if not args.disable_derivative_free_leaf_optim:
            #Update leaves with derivate-free algorithm
            #Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys] #shape (batchsize, num_classes) 
                for leaf in tree.leaves:  
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)  
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                    leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)
        
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))
        # Count the number of correct classifications in EACH CLASS
        y_num = torch.where(ys_pred_max == ys, ys, neg)
        uni_y_list, count_list = torch.unique(ys, return_counts=True)
        for uni_y, count in zip(uni_y_list, count_list):
            y_num_uni = torch.sum(torch.where(y_num == uni_y.expand(y_num.shape), one, zero))
            class_acc[int(uni_y)] += float(torch.true_divide(y_num_uni,count))

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
        )
        # Compute metrics over this batch
        total_loss+=loss.item()
        if isinstance(nor_loss, type(loss)):
            nor_total_loss+=nor_loss.item()
        if isinstance(hcm_loss, type(loss)):
            hcm_total_loss+=hcm_loss.item()
        if isinstance(mixup_loss, type(loss)):
            mixup_total_loss+=mixup_loss.item()
        if isinstance(smoothing_loss, type(loss)):
            smoothing_total_loss+=smoothing_loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)
    for c in range(len(num_class_list)):
        class_acc[c] = class_acc[c]/float(i+1)
    train_info['total_loss'] = total_loss/float(i+1)
    train_info['nor_loss'] = nor_total_loss/float(i+1)
    train_info['hcm_loss'] = hcm_total_loss/float(i+1)
    train_info['mixup_loss'] = mixup_total_loss/float(i+1)
    train_info['smoothing_loss'] = smoothing_total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['class_accuracy'] = class_acc
    return train_info 

# def train_epoch(tree: ProtoTree,
#                 train_loader: DataLoader,
#                 optimizer: torch.optim.Optimizer,
#                 epoch: int,
#                 disable_derivative_free_leaf_optim: bool,
#                 device,
#                 log: Log = None,
#                 log_prefix: str = 'log_train_epochs',
#                 progress_prefix: str = 'Train Epoch'
#                 ) -> dict:
    
#     tree = tree.to(device)
#     # Make sure the model is in eval mode
#     tree.eval()
#     # Store info about the procedure
#     train_info = dict()
#     total_loss = 0.
#     total_acc = 0.
#     # Create a log if required
#     log_loss = f'{log_prefix}_losses'

#     nr_batches = float(len(train_loader))
#     with torch.no_grad():
#         _old_dist_params = dict()
#         for leaf in tree.leaves:
#             _old_dist_params[leaf] = leaf._dist_params.detach().clone()
#         # Optimize class distributions in leafs
#         eye = torch.eye(tree._num_classes).to(device)

#     # Show progress on progress bar
#     train_iter = tqdm(enumerate(train_loader),
#                     total=len(train_loader),
#                     desc=progress_prefix+' %s'%epoch,
#                     ncols=0)
#     # Iterate through the data set to update leaves, prototypes and network
#     for i, (xs, ys) in train_iter:
#         # Make sure the model is in train mode
#         tree.train()
#         # Reset the gradients
#         optimizer.zero_grad()

#         xs, ys = xs.to(device), ys.to(device)

#         # Perform a forward pass through the network
#         ys_pred, info = tree.forward(xs)

#         # Learn prototypes and network with gradient descent. 
#         # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
#         # Compute the loss
#         if tree._log_probabilities:
#             loss = F.nll_loss(ys_pred, ys)
#         else:
#             loss = F.nll_loss(torch.log(ys_pred), ys)
        
#         # Compute the gradient
#         loss.backward()
#         # Update model parameters
#         optimizer.step()
        
#         if not disable_derivative_free_leaf_optim:
#             #Update leaves with derivate-free algorithm
#             #Make sure the tree is in eval mode
#             tree.eval()
#             with torch.no_grad():
#                 target = eye[ys] #shape (batchsize, num_classes) 
#                 for leaf in tree.leaves:  
#                     if tree._log_probabilities:
#                         # log version
#                         update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
#                     else:
#                         update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)  
#                     leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
#                     F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
#                     leaf._dist_params += update

#         # Count the number of correct classifications
#         ys_pred_max = torch.argmax(ys_pred, dim=1)
        
#         correct = torch.sum(torch.eq(ys_pred_max, ys))
#         acc = correct.item() / float(len(xs))

#         train_iter.set_postfix_str(
#             f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
#         )
#         # Compute metrics over this batch
#         total_loss+=loss.item()
#         total_acc+=acc

#         if log is not None:
#             log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

#     train_info['loss'] = total_loss/float(i+1)
#     train_info['train_accuracy'] = total_acc/float(i+1)
#     return train_info 



def train_epoch_kontschieder(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:

    tree = tree.to(device)

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    # Create a log if required
    log_loss = f'{log_prefix}_losses'
    if log is not None and epoch==1:
        log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')
    
    # Reset the gradients
    optimizer.zero_grad()

    if disable_derivative_free_leaf_optim:
        print("WARNING: kontschieder arguments will be ignored when training leaves with gradient descent")
    else:
        if tree._kontschieder_normalization:
            # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
            for _ in range(10):
                # Train leaves with derivative-free algorithm using normalization factor
                train_leaves_epoch(tree, train_loader, epoch, device)
        else:
            # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
            train_leaves_epoch(tree, train_loader, epoch, device)
    # Train prototypes and network. 
    # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)
    # Make sure the model is in train mode
    tree.train()
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Perform a forward pass through the network
        ys_pred, _ = tree.forward(xs)
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)
        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)
        
        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
        )
        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)
        
    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info 

# Updates leaves with derivative-free algorithm
def train_leaves_epoch(tree: ProtoTree,
                        train_loader: DataLoader,
                        epoch: int,
                        device,
                        progress_prefix: str = 'Train Leafs Epoch'
                        ) -> dict:

    #Make sure the tree is in eval mode for updating leafs
    tree.eval()

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

        # Show progress on progress bar
        train_iter = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)
        
        
        # Iterate through the data set
        update_sum = dict()

        # Create empty tensor for each leaf that will be filled with new values
        for leaf in tree.leaves:
            update_sum[leaf] = torch.zeros_like(leaf._dist_params)
        
        for i, (xs, ys) in train_iter:
            xs, ys = xs.to(device), ys.to(device)
            #Train leafs without gradient descent
            out, info = tree.forward(xs)
            target = eye[ys] #shape (batchsize, num_classes) 
            for leaf in tree.leaves:  
                if tree._log_probabilities:
                    # log version
                    update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - out, dim=0))
                else:
                    update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/out, dim=0)
                update_sum[leaf] += update

        for leaf in tree.leaves:
            leaf._dist_params -= leaf._dist_params #set current dist params to zero
            leaf._dist_params += update_sum[leaf] #give dist params new value




# ========= server training ===========
def server_train_all_epoch(tree: ProtoTree,
                    proto_feature: list,
                    proto_y: list,
                    num_class_list: list,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    disable_derivative_free_leaf_optim: bool,
                    args: argparse,
                    # imbalance_loss_flag: int,
                    # : int,
                    device,
                    log: Log = None,
                    log_prefix: str = 'log_train_epochs',
                    progress_prefix: str = 'Server Train Epoch',
                    ) -> dict:
    
    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    # nr_batches = float(len(train_loader))
    nr_batches = float(len(proto_feature))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)
    # print('total prototype feature in this client: ', nr_batches)
    # if args.server_imbloss:
    #     print("!!!!!!!!! Using server_imbloss !!!!!!!!!")
    # Show progress on progress bar
    train_loader = zip(proto_feature, proto_y)
    train_iter = tqdm(enumerate(train_loader),
                    total=len(proto_feature),
                    desc=progress_prefix+' %s'%epoch,
                    position=0, 
                    leave=True,
                    # ncols=0
                    )
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()
        xs = torch.Tensor(xs).to(device, dtype=torch.float32)
        ys = torch.Tensor(ys).to(device, dtype=torch.int64)
        # print(xs.shape)
        # Perform a forward pass through the network
        ys_pred, info = tree.tree_forward(xs)
        # Learn prototypes and network with gradient descent. 
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
            # loss = F.nll_loss(ys_pred, ys) + F.nll_loss(ys_pred * mask, ys)
        else:
            # ys_pred = torch.log(ys_pred) + bsce_weight.unsqueeze(0).expand(ys_pred.shape[0], -1).log()
            loss = F.nll_loss(torch.log(ys_pred), ys)
            # loss = loss*(-1)
            # loss = F.nll_loss(torch.log(ys_pred), ys) + F.nll_loss(torch.log(ys_pred * mask), ys)
        # Compute the gradient
        assert torch.isnan(loss).sum() == 0, log.log_message("先看loss是不是nan,如果loss是nan,那麼說明可能是在forward的過程中出現瞭第一條列舉的除0或者log0的操作. Loss= "+str(loss))
        loss.backward()
        max_norm = 3
        torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm, norm_type=2)
        # for p in tree.parameters():
        #     assert torch.isnan(p).sum() == 0 , log.log_message("Before optimizer tree.parameters(): "+str(p))
        # Update model parameters
        if torch.isnan(tree.prototype_layer.prototype_vectors).any():
            print('Error: tree.prototype_layer.prototype_vectors NaN values in train.py! Before optimizer.step()')
        optimizer.step()
        # for p in tree.parameters():
        #     assert torch.isnan(p).sum() == 0, log.log_message("After optimizer tree.parameters(): "+str(p))
        #     if not isinstance(p.grad, type(None)):
        #         assert torch.isnan(p.grad).sum() == 0, log.log_message("tree.parameters().grad: "+str(p.grad))
        if torch.isnan(tree.prototype_layer.prototype_vectors).any():
            print('Error: tree.prototype_layer.prototype_vectors NaN values in train.py! After optimizer.step()')
        if not disable_derivative_free_leaf_optim:
            #Update leaves with derivate-free algorithm
            #Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys] #shape (batchsize, num_classes) 
                for leaf in tree.leaves:  
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)  
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                    leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)
        
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        # train_iter.set_postfix_str(
        #     f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
        # )
        train_iter.set_postfix_str(f'Loss: {loss.item():.3f}, Acc: {acc:.3f}')
        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc
        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)
    train_iter.close()
    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info 