

from ast import arg
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from Dataset.dataset import show_clients_data_distribution, Indices2Dataset, classify_label, Indices2Dataset_contrastive
from Dataset.data import get_data
from Dataset.sampling import iid
from Dataset.sample_dirichlet import clients_indices

from Model.prototree.prototree import ProtoTree
from Model.prototree.prune import prune
from Model.prototree.project import  project_with_class_constraints
from Model.prototree.upsample import upsample
from Model.prototree.test import eval, eval_fidelity

from util.net import get_network
from util.fed_api import Global, Local
from util.info_tool import load_ckpt
from util.options import save_args, args_parser
from util.visualize import gen_vis
from util.analyse import *


from copy import deepcopy
import numpy as np
import random
import sys
import datetime
import pickle
import os 

def main():
    args = args_parser()
    checkpoint, args, save_dir = load_ckpt(args)
    # sys.exit()
    random_state = np.random.RandomState(args.seed)
    # ----- Load data -----
    data_local_training, data_local_projecting, data_global_test, img_num_list, classes = get_data(args)
    num_classes = len(np.unique(data_global_test.targets))
    args.num_classes = num_classes
    # test_label = [0 for _ in range(num_classes)]
    # for l in data_global_test.targets:
    #     test_label[l]+=1
    # ----- Distribute data -----
    # dict_trainset_idx, _, _, original_dict_per_client = imbalance_iidTrain_iidTest(data_local_training, data_local_training, data_global_test, args.num_clients, args.imb_factor)
    if not args.test_data == '':
        with open(args.test_data, "rb") as f:
            list_client2indices = pickle.load(f)
        print("Load data distribution by ", args.test_data)
    elif args.iid:
        # ------- iid -------
        dict_trainset_idx, _, _, original_dict_per_client = iid(data_local_training, data_local_training, data_global_test, args.num_clients)
        list_client2indices = [dict_trainset_idx[k] for k in dict_trainset_idx]
    else:
        # ------- non-iid -------
        list_label2indices_train_new = classify_label(data_local_training, num_classes)
        list_client2indices = clients_indices(deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, args.seed)                                     
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes, None)
    # print("original_dict_per_client", original_dict_per_client)
    # sys.exit()
    testloader = DataLoader(data_global_test, args.batch_size_test, 
                        num_workers=args.workers,
                        pin_memory=True)

    indices2data = Indices2Dataset(data_local_projecting)
    online_clients = range(args.num_online_clients)
    for client in online_clients:
        indices2data.load(list_client2indices[client])
        # indices2data.load_imgs()
        projectloader = DataLoader(
                        indices2data,
                        batch_size=args.batch_size_local_training, shuffle=False,
                        pin_memory=True, num_workers=args.workers)

    # print(projectloader.dataset)
    # print(projectloader.dataset.dataset.data[0].shape)
    # print(classes)
    # sys.exit()
    # ----- Setting Device ----- 
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    args.device = device

    # ----- Create Model -----
    feature_net, add_on_layers, server_set, client_set = get_network(args)
    classifier_server = server_set["cls"]
    criterion_server = server_set["loss"]
    prototree = ProtoTree(feature_net, add_on_layers, classifier_server)
    
    # ----- Load Model -----
    prototree.load_state_dict(checkpoint['model_state_dict'])
    # args_dict = checkpoint['args_dict']

        
    '''
        PRUNE
    '''
    pruned = prune(prototree, args.pruning_threshold_leaves)
    name = "pruned"
    pruned_tree = deepcopy(prototree)
    # Analyse and evaluate pruned prototree
    # leaf_labels = analyse_leafs(prototree, epoch+2, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    # analyse_leaf_distributions(prototree, log)
    eval_info = eval(prototree, testloader, device, num_classes)
    pruned_test_acc = eval_info['test_accuracy']
    pruned_tree = prototree

    '''
        PROJECT
    '''
    project_info, prototree = project_with_class_constraints(deepcopy(pruned_tree), projectloader, device, args)
    name = "pruned_and_projected"
    pruned_projected_tree = deepcopy(prototree)
    # # --- Analyse and evaluate pruned prototree with projected prototypes ---
    # average_distance_nearest_image(project_info, prototree)
    # eval_info = eval(prototree, testloader, device, num_classes)
    # pruned_projected_test_acc = eval_info['test_accuracy']
    # eval_info_samplemax = eval(prototree, testloader, device, num_classes, 'sample_max')
    # get_avg_path_length(prototree, eval_info_samplemax)
    # eval_info_greedy = eval(prototree, testloader, device, num_classes, 'greedy')
    # get_avg_path_length(prototree, eval_info_greedy)
    # fidelity_info = eval_fidelity(prototree, testloader, device)

    # Upsample prototype for visualization
    # project_info = upsample(prototree, project_info, projectloader, name, args, save_dir)
    project_info = upsample(pruned_projected_tree, project_info, projectloader, name, args, save_dir)
    # visualize prototree
    gen_vis(pruned_projected_tree, name, args, classes, save_dir)





if __name__ == '__main__':
    start= datetime.datetime.now()
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    # mp.set_start_method('spawn')
    main()
    end = datetime.datetime.now()
    print("ALL Execute time: " ,(end - start))


