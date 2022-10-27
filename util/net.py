import argparse
from ast import arg
from copy import deepcopy
from http import client
from locale import normalize
from statistics import mean
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F
from torch.autograd import Variable

from Model.prototree.prototree import ProtoTree, Tree
from Model.features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from Model.features.resnet_feature_cifar import resnet32_fe
from Model.features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from Model.features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,vgg19_features, vgg19_bn_features
from Model.features.resnet8 import resnet8
from util.info_tool import write_info

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features,
                                 'resnet32' : resnet32_fe,
                                 'resnet8' : resnet8}

"""
    Create network with pretrained features and 1x1 convolutional layer

"""


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-0.08 , 0.08)
        m.bias.data.fill_(0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

# def init_weights_xavier(m):
#     if type(m) == torch.nn.Conv2d:
#         torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='sigmoid')

def init_weights_relu(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')

class add_on(nn.Module):
    def __init__(self, input_channels, num_features, kernel_size, device):
        super(add_on, self).__init__()
        self.num_features = num_features
        self.input_channels = input_channels
        self.ker_size = kernel_size
        self.device = device
        self.conv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.num_features, kernel_size=1, bias=False)
        self.act_f1 = nn.Sigmoid()
        with torch.no_grad():
            self.conv.apply(init_weights_xavier)

       
    def forward(self, x):
        # global_feat = self.add_on_layers(x)  # (b, num_features=128, ker_size=(7or8), ker_size)
        global_feat = self.conv(x)  # (b, num_features=128, ker_size=(7or8), ker_size)
        global_feat = self.act_f1(global_feat) 
        return global_feat


def get_network(args: argparse.Namespace):
    # Define a conv net for estimating the probabilities at each decision node
    write_info("User ", args.net, " as feature extractor")
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)            
    features_name = str(args.net).upper()
    if features_name.startswith('VGG') or features_name.startswith('RES'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    elif features_name.startswith('DENSE'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
    else:
        raise Exception('other base base_architecture NOT implemented')

    add_on_layers = add_on(first_add_on_layer_in_channels, args.num_features, args.ker_size, args.device)
    classify_layer_in_channels = \
        [i for i in add_on_layers.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    classifier_linear = nn.Sequential(
                    nn.AvgPool2d(kernel_size=args.ker_size),
                    nn.Flatten(),
                    nn.Linear(
                        in_features=classify_layer_in_channels,
                        out_features=args.num_classes,
                    )
                ) 
    write_info("classify_layer_in_channels: ", classify_layer_in_channels)
    write_info("num_classes: ", args.num_classes)

    classifier_tree = Tree(args)  
    # --- init tree ---
    mean = 0.5
    std = 0.1
    with torch.no_grad():
        torch.nn.init.normal_(classifier_tree.prototype_layer.prototype_vectors, mean=mean, std=std)


    criterion_linear = CrossEntropyLoss()
    criterion_tree = NLLLoss()
    server = {}
    client = {}
    if args.server_classifier == "tree":
        write_info("Server Classifier : Tree")
        server["cls"] = deepcopy(classifier_tree)
        server["loss"] = deepcopy(NLLLoss())
        # server["loss"] = deepcopy(criterion_tree)
    else:
        write_info("Server Classifier : Linear")
        server["cls"] = deepcopy(classifier_linear)
        server["loss"] = deepcopy(CrossEntropyLoss())
        # server["loss"] = deepcopy(criterion_linear)
    if args.client_classifier == "tree":
        write_info("Client Classifier : Tree")
        client["cls"] = deepcopy(classifier_tree)
        client["loss"] = deepcopy(criterion_tree)
    else:
        write_info("Client Classifier : Linear")
        client["cls"] = deepcopy(classifier_linear)
        client["loss"] = deepcopy(criterion_linear)
    
    return features, add_on_layers, server, client

def get_optimizer(args, cls_type, feature_net, add_on_layers, classifier):
    if cls_type == "tree":
        
        dist_params = []
        params_to_freeze = []
        params_to_train = []
        for name,param in classifier.named_parameters():
            if 'dist_params' in name:
                dist_params.append(param)
        if 'resnet50' in args.net: 
            # freeze resnet50 except last convolutional layer
            for name,param in feature_net.named_parameters():
                if 'layer4.2' not in name:
                    params_to_freeze.append(param)
                else:
                    params_to_train.append(param)
            if cls_type == 'SGD':
                paramlist = [
                    {"params": params_to_freeze, "lr": args.lr_net, 'initial_lr': args.lr_net, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                    {"params": params_to_train, "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum}, 
                    {"params": add_on_layers.parameters(), "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
                    {"params": classifier.prototype_layer.parameters(), "lr": args.lr, 'initial_lr': args.lr,"weight_decay_rate": 0,"momentum": 0}]

                if args.disable_derivative_free_leaf_optim:
                    paramlist.append({"params": dist_params, "lr": args.lr_pi, 'initial_lr': args.lr_pi, "weight_decay_rate": 0})
            else:
                paramlist = [
                    {"params": params_to_freeze, "lr": args.lr_net, 'initial_lr': args.lr_net, "weight_decay_rate": args.weight_decay},
                    {"params": params_to_train, "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay}, 
                    {"params": add_on_layers.parameters(), "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay},
                    {"params": classifier.prototype_layer.parameters(), "lr": args.lr, 'initial_lr': args.lr,"weight_decay_rate": 0}]

                if args.disable_derivative_free_leaf_optim:
                    paramlist.append({"params": dist_params, "lr": args.lr_pi, 'initial_lr': args.lr_pi, "weight_decay_rate": 0})

        else:
            for name,param in feature_net.named_parameters():
                params_to_freeze.append(param)
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, 'initial_lr': args.lr_net, "weight_decay_rate": args.weight_decay},
                {"params": add_on_layers.parameters(), "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": classifier.prototype_layer.parameters(), "lr": args.lr, 'initial_lr': args.lr,"weight_decay_rate": 0}]
            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, 'initial_lr': args.lr_pi, "weight_decay_rate": 0})

    else:
        if args.optimizer == 'SGD':
            paramlist = [
                {"params": feature_net.parameters(), "lr": args.lr_net, 'initial_lr': args.lr_net, "weight_decay_rate": args.weight_decay,"momentum": args.momentum}, 
                {"params": add_on_layers.parameters(), "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
                {"params": classifier.parameters(), "lr": args.lr, 'initial_lr': args.lr,"weight_decay_rate": 0,"momentum": 0}]

        else:
            paramlist = [
                {"params": feature_net.parameters(), "lr": args.lr_net, 'initial_lr': args.lr_net, "weight_decay_rate": args.weight_decay},
                {"params": add_on_layers.parameters(), "lr": args.lr_block, 'initial_lr': args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": classifier.parameters(), "lr": args.lr, 'initial_lr': args.lr,"weight_decay_rate": 0}]
            
    if args.optimizer == 'SGD':
        optimizer =  torch.optim.SGD(paramlist,
                            lr=args.lr,
                            momentum=args.momentum)
    if args.optimizer == 'Adam':
        optimizer =  torch.optim.Adam(paramlist,lr=args.lr,eps=1e-07)
    if args.optimizer == 'AdamW':
        optimizer =  torch.optim.AdamW(paramlist,lr=args.lr,eps=1e-07, weight_decay=args.weight_decay)
    return optimizer


