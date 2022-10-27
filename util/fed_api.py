from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import eq, no_grad
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from Dataset.dataset import ServerDataset
from Dataset.sampler import ClassAwareSampler
from util.acc_calculate import accuracy
from util.net import get_optimizer
from util.info_tool import write_info

import numpy as np
import copy
import sys

class Global(object):
    def __init__(self,
                 args,
                 writer,
                 criterion_server, 
                 criterion_client, 
                 shot_dict,
                 feature_net: nn.Module = nn.Identity(),
                 add_on_layers: nn.Module = nn.Identity(),  
                 classifier_client: nn.Module = nn.Identity(),
                 classifier_server: nn.Module = nn.Identity(),
                 ):
        self.shot_dict = shot_dict
        self.device = args.device
        self.args = args
        self.num_classes = args.num_classes
        self.writer = writer

        self.criterion_server = criterion_server
        self.criterion_client = criterion_client
        self.feature_net = feature_net
        self.add_on_layers = add_on_layers
        self.classifier_server = classifier_server
        self.classifier_client = classifier_client

        self.feature_net_fed_weight = copy.deepcopy(feature_net.state_dict())
        self.add_on_layers_fed_weight = copy.deepcopy(add_on_layers.state_dict())
        self.classifier_client_fed_weight = copy.deepcopy(classifier_client.state_dict())

        self.optimizer = get_optimizer(args, args.server_classifier, self.feature_net, self.add_on_layers, self.classifier_server)
        milestones = []
        ratio = (args.num_rounds*args.global_epoch)/args.milestones[-1]
        for m in args.milestones:
            mil = ratio*m
            milestones.append(mil)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=args.gamma)

        self.round = 1
        self.weight_update_flag = True
        self.round_client_data_num = 0.

        self.cls_num_list = []
        self.labels=[]
        self.samples_per_cls=[]
        self.net_weight_list = []
        self.add_weight_list = []
        self.cls_weight_list = []
        self.soft = nn.Softmax(dim=-1).to(self.device)

    def global_eval(self, data_test, batch_size_test, cls_part="server", commit=""):
        write_info("Evaluate ",cls_part)
        if cls_part == "server":
            cls = self.classifier_server
        else:
            cls = self.classifier_client
        self.feature_net.eval()
        self.add_on_layers.eval()
        cls.eval()
        self.feature_net = self.feature_net.to(self.device)
        self.add_on_layers = self.add_on_layers.to(self.device)
        cls = cls.to(self.device)
        # write_info("self.args.num_classes: ", self.args.num_classes)
        pre_list = [0 for _ in range(self.args.num_classes)]
        cor_list = [0 for _ in range(self.args.num_classes)]
        lab_list = [0 for _ in range(self.args.num_classes)]
        test_loader = DataLoader(data_test, batch_size_test, 
                                    num_workers=self.args.workers,
                                    pin_memory=True)
        with no_grad():
            acc1_avg = 0
            acc3_avg = 0 
            for images, labels in tqdm(test_loader, desc='Eval'):
            # for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.feature_net(images)
                features = self.add_on_layers(features)
                if cls_part == "server":
                    if self.args.server_classifier == "tree":
                        outputs, _  = cls(features)
                    else:
                        outputs = cls(features)
                else:
                    if self.args.client_classifier == "tree":
                        outputs, _  = cls(features)
                    else:
                        outputs = cls(features)
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
            predict_dict = {str(c):num for c, num in enumerate(pre_list)}
            correct_dict = {str(c):num for c, num in enumerate(cor_list)}
            label_dict = {str(c):num for c, num in enumerate(lab_list)}
            correct_ratio_dict={}
            correct_ratio={"few":0,"medium":0,"many":0}
            correct_ratio_num={"few":0,"medium":0,"many":0}
            for c, (cor, lab) in enumerate(zip(cor_list, lab_list)):
                if (not lab==0): 
                    correct_ratio_dict[str(c)] = float(cor)/float(lab)*100
                    correct_ratio_num[self.shot_dict[c]]+=1
                    correct_ratio[self.shot_dict[c]]+=float(cor)/float(lab)*100
                else :
                    correct_ratio_dict[str(c)] = -1 
            # print(correct_ratio_num)
            correct_ratio["many"]=correct_ratio["many"]/correct_ratio_num["many"] if not correct_ratio_num["many"]==0 else 0
            correct_ratio["few"]=correct_ratio["few"]/correct_ratio_num["few"] if not correct_ratio_num["few"]==0 else 0
            correct_ratio["medium"]=correct_ratio["medium"]/correct_ratio_num["medium"] if not correct_ratio_num["medium"]==0 else 0


        print(cls_part+" Eval Acc1: ", acc1_avg.item())
        self.writer.add_scalar(cls_part+commit+'_Eval/acc1/', acc1_avg, self.round)
        self.writer.add_scalar(cls_part+commit+'_Eval/acc3/', acc3_avg, self.round)
        self.writer.add_scalars(cls_part+commit+'_Eval/Correct_Ratio/', correct_ratio, self.round)
        if self.args.detailed_eval:
            print(cls_part+" correct_ratio: ", correct_ratio)
            print(cls_part+" Correct_Ratio_per_class: ", correct_ratio_dict)
            self.writer.add_scalars(cls_part+commit+'_Eval/Predict_num_per_class/', predict_dict, self.round)
            self.writer.add_scalars(cls_part+commit+'_Eval/Correct_num_per_class/', correct_dict, self.round)
            self.writer.add_scalars(cls_part+commit+'_Eval/Label_num_per_class/', label_dict, self.round)
            self.writer.add_scalars(cls_part+commit+'_Eval/Correct_Ratio_per_class/', correct_ratio_dict, self.round)
        self.feature_net = self.feature_net.to('cpu')
        self.add_on_layers = self.add_on_layers.to('cpu')
        cls = cls.to('cpu')
        torch.cuda.empty_cache()

    def train_classifier(self, feature_list:list, label_list:list):
        train_dataset = ServerDataset(feature_list, label_list)
        # ---- reasample dataset if wanted ----
        if self.args.server_resampler:
            shuffle_flag = False
            balance_sampler = ClassAwareSampler(train_dataset, self.args.max_sample_proportion)
            server_sampler = balance_sampler
        else:
            shuffle_flag = True
            server_sampler = None
        train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size_global_training, shuffle=shuffle_flag,
                pin_memory=True, sampler=server_sampler, num_workers=self.args.workers)
        self.classifier_server = self.classifier_server.to(self.device)

        data_count = [0 for _ in range(self.args.num_classes)]
        for (_, ys) in train_loader:
            for y in ys:
                data_count[int(y)] += 1
        data_count_dict = {str(c):num for c, num in enumerate(data_count)}
        self.writer.add_scalars('Distribution/Server_train', data_count_dict, self.round)
        nr_batches = float(len(train_loader))

        # for epoch in range(self.args.global_epoch):
        for epoch in tqdm(range(self.args.global_epoch), desc='ServerTrain'):
            # if epoch%10==0:
                # print("Server train epoch", epoch)
            self.classifier_server.eval()
            if self.args.server_classifier == "tree":
                with torch.no_grad():
                    _old_dist_params = dict()
                    for leaf in self.classifier_server.leaves:
                        _old_dist_params[leaf] = leaf._dist_params.detach().clone()
                    # ---- Optimize class distributions in leafs ----
                    eye = torch.eye(self.classifier_server._num_classes).to(self.device)
            pre_list = [0 for _ in range(self.args.num_classes)]
            cor_list = [0 for _ in range(self.args.num_classes)]
            lab_list = [0 for _ in range(self.args.num_classes)]
            acc1_avg = 0
            acc3_avg = 0 
            for features, labels in train_loader:
            # for features, labels in tqdm(train_loader, desc='ServerTrain'):
                self.classifier_server.train()
                self.optimizer.zero_grad()
                features, labels = features.to(self.device), labels.to(self.device)
                if self.args.server_classifier == "tree":
                    outputs, info = self.classifier_server(features)
                else:
                    outputs = self.classifier_server(features)
                # ---- Compute the loss ----
                loss_net = self.criterion_server(outputs, labels)
                # ---- Compute the gradient ----
                loss_net.backward()
                # ---- Update model parameters ----
                self.optimizer.step()
                # ---- Learn prototypes and network with gradient descent. 
                # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well. ----
                if self.args.server_classifier == "tree":
                    if not self.args.disable_derivative_free_leaf_optim:
                        #Update leaves with derivate-free algorithm
                        #Make sure the tree is in eval mode
                        self.classifier_server.eval()
                        with torch.no_grad():
                            target = eye[labels] #shape (batchsize, num_classes) 
                            for leaf in self.classifier_server.leaves:  
                                if self.classifier_server._log_probabilities:
                                    # log version
                                    update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - outputs, dim=0))
                                else:
                                    update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/outputs, dim=0)  
                                leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                                F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                                leaf._dist_params += update
                
                acc1, acc3 = accuracy(outputs, labels, topk=(1, 3))
                acc1_avg += acc1
                acc3_avg += acc3
                outputs_pred_max = torch.argmax(outputs, dim=1)
                for (pre, cor, lab) in zip(outputs_pred_max, eq(outputs_pred_max.cpu(), labels.cpu()), labels):
                    pre_list[pre]+=1
                    if cor:
                        cor_list[pre]+=1
                    lab_list[lab] += 1

            self.scheduler.step()    
            # ---- calculate accuarcy and write in tensorboard ----
            acc1_avg = acc1_avg / float(len(train_loader))
            acc3_avg = acc3_avg / float(len(train_loader))

            writer_epoch = (self.round-1)*self.args.global_epoch + epoch
            self.writer.add_scalar('Server_train/acc1', acc1_avg, writer_epoch)

        self.classifier_server = self.classifier_server.to('cpu')
        torch.cuda.empty_cache()

    def next_round(self):
        self.round += 1

    def update_client_classifier(self):
        if self.args.server_classifier == self.args.client_classifier:
            self.classifier_client.load_state_dict(copy.deepcopy(self.classifier_server.state_dict()))

    def weight_collect_weighted(self, feature_layer_weight, add_on_layer_weight, classifier_weight, cls_num_list):
        self.net_weight_list.append(feature_layer_weight)
        self.add_weight_list.append(add_on_layer_weight)
        self.cls_weight_list.append(classifier_weight)
        self.cls_num_list.append(cls_num_list)

    def weight_update_weighted_test(self):
        cls_num_list = torch.tensor(self.cls_num_list).float()
        if not (self.args.num_clients==self.args.num_online_clients and self.round > 1):
            # ----- number of data -----
            nums_local_data = torch.sum(cls_num_list, dim=1)
            nums_local_data_mean = torch.div(nums_local_data, torch.sum(nums_local_data))
            # ----- number of nonzero classses -----
            cls_nonzero = ~torch.eq(cls_num_list,0)
            num_class = torch.sum(cls_nonzero, dim=1).float()
            class_num_per_cli = torch.div(num_class, self.num_classes)
            # ----- rare ratio -----
            num_per_class = torch.sum(cls_num_list,dim=0)
            num_all = torch.sum(num_per_class)
            rare_per_class = 1-torch.div(num_per_class,num_all)
            rare_per_client = torch.sum(torch.mul(rare_per_class,cls_nonzero), dim=1)
            rare_per_client = torch.div(rare_per_client, class_num_per_cli)

            # ----- weighted by hyperparameters -----
            all_mul = 1
            if self.args.lamda_cls_num : 
                class_num_soft = class_num_per_cli*self.args.lamda_cls_num
                all_mul = torch.mul(all_mul, class_num_soft)
            if self.args.lamda_data_num : 
                nums_local_data_soft = nums_local_data_mean*self.args.lamda_data_num
                all_mul = torch.mul(all_mul, nums_local_data_soft)
            if self.args.lamda_data_rare:
                rare_data_soft = rare_per_client*self.args.lamda_data_rare
                all_mul = torch.mul(all_mul, rare_data_soft)
            weighted = self.soft(all_mul)
            self.weighted = weighted

        # ----- update global weight -----
        num_client = len(cls_num_list)
        for i in range(num_client):
            # add client weight 
            if i == 0:
                for k in self.feature_net_fed_weight.keys():
                    self.feature_net_fed_weight[k] = self.net_weight_list[i][k]*self.weighted[i]
                for k in self.add_on_layers_fed_weight.keys():
                    self.add_on_layers_fed_weight[k] = self.add_weight_list[i][k]*self.weighted[i]
                for k in self.classifier_client_fed_weight.keys():
                    self.classifier_client_fed_weight[k] = self.cls_weight_list[i][k]*self.weighted[i]
            else:
                for k in self.feature_net_fed_weight.keys():
                    self.feature_net_fed_weight[k] += self.net_weight_list[i][k]*self.weighted[i]
                for k in self.add_on_layers_fed_weight.keys():
                    self.add_on_layers_fed_weight[k] += self.add_weight_list[i][k]*self.weighted[i]
                for k in self.classifier_client_fed_weight.keys():
                    self.classifier_client_fed_weight[k] += self.cls_weight_list[i][k]*self.weighted[i]
                
        self.feature_net.load_state_dict(self.feature_net_fed_weight)
        self.add_on_layers.load_state_dict(self.add_on_layers_fed_weight)
        self.classifier_client.load_state_dict(self.classifier_client_fed_weight)
        self.cls_num_list.clear()
        self.net_weight_list.clear()
        self.add_weight_list.clear()
        self.cls_weight_list.clear()

        # ------ the number of additional data client needs to upload ------
        num_per_class = torch.sum(cls_num_list,dim=0)
        self.samples_per_cls = num_per_class
        cls_num_list = cls_num_list.float()


class Local(object):
    def __init__(self,
                 data_client,
                 cls_num_list: int,
                 args,
                 criterion, 
                 feature_net: nn.Module = nn.Identity(),
                 add_on_layers: nn.Module = nn.Identity(),  
                 classifier: nn.Module = nn.Identity(),
                 last_epoch: int=-1
                 ):
        self.args = args
        # ------ dataset ------
        self.data_client = data_client
        if self.args.client_resampler:
            shuffle_flag = False
            self.data_client.load_targets()
            balance_sampler = ClassAwareSampler(self.data_client)
            client_sampler = balance_sampler
        else:
            shuffle_flag = True
            client_sampler = None
        self.data_loader = DataLoader(
                self.data_client,
                batch_size=self.args.batch_size_local_training, shuffle=shuffle_flag,
                pin_memory=True, sampler=client_sampler, num_workers=self.args.workers)

        # ------ Model and Losses ------
        self.device = args.device
        self.criterion = criterion.to(self.device)
        self.feature_net = feature_net.to(self.device)
        self.add_on_layers = add_on_layers.to(self.device)
        self.classifier = classifier.to(self.device)
        self.class_compose = torch.tensor(cls_num_list).float().to(self.device)

        # ------ Optimizer and Scheduler ------
        step_opt=1
        lr_ep = last_epoch
        if lr_ep == 0: # init
            lr_ep=-1
        milestones = []
        ratio = (args.num_rounds*args.local_epoch)/args.milestones[-1]
        for m in args.milestones:
            # mil = int(ratio*m)
            mil = ratio*m
            milestones.append(mil)
            if lr_ep > mil:
                step_opt += 1
        self.optimizer = get_optimizer(self.args, self.args.client_classifier, self.feature_net, self.add_on_layers, self.classifier)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=self.args.gamma, last_epoch=lr_ep)

        self.last_epoch = last_epoch

    def local_train(self):

        nr_batches = float(len(self.data_loader))
        loss_list = ["total","normal"]

        for ep in range(1, self.args.local_epoch+1):

            total_loss = {l:0. for l in loss_list }
            self.feature_net.train()
            self.add_on_layers.train()
            self.classifier.eval()
            if self.args.client_classifier == "tree":
                with torch.no_grad():
                    _old_dist_params = dict()
                    for leaf in self.classifier.leaves:
                        _old_dist_params[leaf] = leaf._dist_params.detach().clone()
                    # Optimize class distributions in leafs
                    eye = torch.eye(self.classifier._num_classes).to(self.device)

            for images, labels in self.data_loader:
                self.optimizer.zero_grad()
                self.classifier.train()
                images, labels = images.to(self.device), labels.to(self.device)

                if self.last_epoch+ep <= self.args.freeze_epoch:
                    with torch.no_grad():
                        features = self.feature_net(images)
                else:
                    features = self.feature_net(images)
                features = self.add_on_layers(features)
                if self.args.client_classifier == "tree":
                    outputs, info  = self.classifier(features)
                else:
                    outputs = self.classifier(features)

                ori_loss = self.criterion(outputs, labels)
                total_loss["normal"]+=ori_loss.item()
                loss = ori_loss
                    
                loss.backward()
                self.optimizer.step()
                total_loss["total"]+=loss.item()
                
                if self.args.client_classifier == "tree":
                    if not self.args.disable_derivative_free_leaf_optim:
                        #Update leaves with derivate-free algorithm
                        #Make sure the tree is in eval mode
                        self.classifier.eval()
                        with torch.no_grad():
                            target = eye[labels] #shape (batchsize, num_classes) 
                            for leaf in self.classifier.leaves:  
                                if self.classifier._log_probabilities:
                                    # log version
                                    update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - outputs, dim=0))
                                else:
                                    update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/outputs, dim=0)  
                                leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                                F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                                leaf._dist_params += update
            self.scheduler.step()    

    def feature_extract(self):
        self.feature_net.eval()
        self.add_on_layers.eval()
        feature_list = []
        label_list = []
        with no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.feature_net(images)
                features = self.add_on_layers(features)
                feature_list.append(features)
                label_list.append(labels)
        feature_list = torch.cat(feature_list, dim=0)
        label_list = torch.cat(label_list)
        return feature_list, label_list
    
    def upload_params(self):
        return copy.deepcopy(self.feature_net.state_dict()),\
                copy.deepcopy(self.add_on_layers.state_dict()),\
                copy.deepcopy(self.classifier.state_dict()),

    def to_cpu(self): 
        self.criterion = self.criterion.to('cpu')
        self.feature_net = self.feature_net.to('cpu')
        self.add_on_layers = self.add_on_layers.to('cpu')
        self.classifier = self.classifier.to('cpu')
        self.class_compose = self.class_compose.to('cpu')
        torch.cuda.empty_cache()
