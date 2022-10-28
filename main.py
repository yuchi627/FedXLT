

from ast import arg
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Dataset.dataset import show_clients_data_distribution, Indices2Dataset, classify_label
from Dataset.data import get_data
from Dataset.sampling import iid
from Dataset.sample_dirichlet import clients_indices
from Model.prototree.prototree import ProtoTree
from util.net import get_network
from util.fed_api import Global, Local
from util.info_tool import write_info, create_name
from util.options import args_parser, save_args

from copy import deepcopy
import numpy as np
import random
import sys
import datetime
import pickle

def main():
    # ============== Prepare Training data ==============
    args = args_parser()
    log_name = create_name(args)
    tensorboard_dir = "summary/"
    tensorboard_name = tensorboard_dir+log_name
    writer = SummaryWriter(tensorboard_name)
    write_info("Write Summary at : ", tensorboard_name)
    # ----- Load data -----
    data_local_training, _, data_global_test, img_num_list, _ = get_data(args)
    num_classes = len(np.unique(data_global_test.targets))
    args.num_classes = num_classes
    save_args(args, tensorboard_name)    
    shot_dict = {}
    for class_idx, num in enumerate(img_num_list):
        writer.add_scalar('Distribution/Client_total', num, class_idx)
        if num>args.thres_many:
            shot_dict[class_idx] = "many"
        elif num<args.thres_few:
            shot_dict[class_idx] = "few"
        else:
            shot_dict[class_idx] = "medium"
            
    sorted_img_num_list = sorted(img_num_list, reverse = True)
    for class_idx, num_cls in enumerate(sorted_img_num_list):
        writer.add_scalar('Distribution/total_sorted', num_cls, class_idx)  
    test_label = [0 for _ in range(num_classes)]
    for l in data_global_test.targets:
        test_label[l]+=1
    for class_idx, num in enumerate(test_label):
        writer.add_scalar('Distribution/Test', num, class_idx)
    # ----- Distribute data -----
    if args.test_data: # use specify data
        write_info("Load data distribution by ", args.test_data)
        with open(args.test_data, "rb") as f:
            list_client2indices = pickle.load(f)
    elif args.iid or args.num_clients == 1: # iid 
        dict_trainset_idx, _, _, original_dict_per_client = iid(data_local_training, data_local_training, data_global_test, args.num_clients)
        list_client2indices = [dict_trainset_idx[k] for k in dict_trainset_idx]
    else: # non-iid 
        list_label2indices_train_new = classify_label(data_local_training, num_classes)
        list_client2indices = clients_indices(deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, args.seed)                                     
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes, writer)
    # ----- Setting Device ----- 
    if torch.cuda.is_available(): # gpu
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        write_info("Using device: ", 'cuda:{}'.format(torch.cuda.current_device()))
    else: # cpu
        device = torch.device('cpu')
        write_info("Using device: cpu")
    args.device = device
    # ----- Create Model -----
    feature_net, add_on_layers, server_set, client_set = get_network(args)
    classifier_server = server_set["cls"]
    criterion_server = server_set["loss"]
    classifier_client = client_set["cls"]
    criterion_client = client_set["loss"]
    # ----- Create Global Model -----
    global_model = Global(args,
                        writer,
                        criterion_server=criterion_server, 
                        criterion_client=criterion_client, 
                        shot_dict = shot_dict,
                        feature_net=feature_net,  
                        add_on_layers=add_on_layers,
                        classifier_client=classifier_client,
                        classifier_server=classifier_server,
                        )

    indices2data = Indices2Dataset(data_local_training)
    # ============== Start Training ==============
    # for train_round in tqdm(range(1, args.num_rounds+1), desc='Round'):
    for train_round in range(1, args.num_rounds+1):
        print("Round ", train_round)
        online_clients = range(args.num_online_clients)
        #  ----- local training  -----
        for client in tqdm(online_clients, desc='Client'):
        # for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            # ----- Create Local Model -----
            local_model = Local(data_client=data_client,
                                cls_num_list=original_dict_per_client[client], 
                                args=args,
                                criterion=criterion_client,
                                feature_net=deepcopy(global_model.feature_net), 
                                add_on_layers=deepcopy(global_model.add_on_layers), 
                                classifier=deepcopy(global_model.classifier_client), 
                                last_epoch=args.local_epoch*(train_round-1)
                                )
            # ----- Local update ----- 
            local_model.local_train()
            net_params, addon_params, classifier_params = local_model.upload_params() 
            global_model.weight_collect_weighted(net_params, addon_params, classifier_params, original_dict_per_client[client])
            local_model.to_cpu()
        # ----- Aggregate local models with FedAvg -----
        global_model.weight_update_weighted_test()

        if not args.fed_method == "gkt":
            global_model.global_eval(data_global_test, args.batch_size_test, "client")
        if args.fed_method == "gkt":
            global_feature = []
            global_labels = []
            # ----- Use Aggregated model to extract feature -----
            for i, client in enumerate(online_clients):
                indices2data.load(list_client2indices[client])
                data_client = indices2data
                # ----- Create Local Model -----
                local_model = Local(data_client=data_client,
                                cls_num_list=original_dict_per_client[client], 
                                args=args,
                                criterion=criterion_client,
                                feature_net=deepcopy(global_model.feature_net), 
                                add_on_layers=deepcopy(global_model.add_on_layers), 
                                classifier=deepcopy(global_model.classifier_client), 
                                last_epoch=args.local_epoch*(train_round-1)
                                )
                # ----- Extract feature -----
                feature_list, label_list = local_model.feature_extract()
                local_model.to_cpu()

                if args.check_size:
                    with open("data_dis", "wb") as fp:   #Pickling
                        pickle.dump(original_dict_per_client[client], fp)
                    torch.save(feature_list,"feature_list.pt")
                    torch.save(label_list,"label_list.pt")
                global_feature.append(feature_list)
                global_labels.append(label_list)
            

            # ----- Use all client's feature to train global classifier -----
            global_feature = torch.cat(global_feature, dim=0).to('cpu')
            global_labels = torch.cat(global_labels).to('cpu')
            # write_info("global_feature.is_cuda: ", global_feature.is_cuda)
            global_model.train_classifier(global_feature, global_labels)

            # ----- Update client classifier with global trained classifier -----
            if args.cls_ser2cli:
                global_model.update_client_classifier()

            # ----- global eval -----
            global_model.global_eval(data_global_test, args.batch_size_test)
        global_model.next_round()
    writer.close()
    prototree = ProtoTree(global_model.feature_net, global_model.add_on_layers, global_model.classifier_server)
    if args.save_ckpt:
        ckpt_path = "ckpt/"+log_name+".pt"
        torch.save({
            'model_state_dict': prototree.state_dict(),
            'args_dict':args,
            'original_dict_per_client':original_dict_per_client
            }, ckpt_path)
    # ----- write_info some info of this process -----
    done_msg=log_name+" Done !!!!!!"
    write_info(f"\033[0;30;43m{done_msg}\033[0m")

if __name__ == '__main__':
    start= datetime.datetime.now()
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    main()
    end = datetime.datetime.now()
    write_info("ALL Execute time: " ,(end - start))


