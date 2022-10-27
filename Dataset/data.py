import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from copy import deepcopy
import sys
import numpy as np
import argparse
import os

from util.info_tool import write_info
from Dataset.sampler import TwoCropTransform
from Dataset.dataset import classify_label, Indices2Dataset, COVID_Dataset, places365
from Dataset.long_tailed_cifar10 import train_long_tail

def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if 'place' in args.dataset.lower():
        return get_place365(True, args)
    elif 'covid' in args.dataset.lower():
        return get_covid(True, args)
    elif 'cifar100' in args.dataset.lower():
        return get_cifar100(True, args)
    elif 'cifar10' in args.dataset.lower():
        return get_cifar10(True, args)
    else:
        raise Exception(f'Could not load data set "{args.dataset}"!')

# https://github.com/facebookresearch/classifier-balancing
def get_place365(augment: bool, args: argparse.Namespace): 
    num_classes = 8142
    write_info(' ----  Dataset:  Place365 ---- ')
    if args.path_place == "":
        place_dir = "data/place"
    else:
        place_dir = args.path_place
    nor_mean = (0.485, 0.456, 0.406)
    nor_std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=nor_mean,std=nor_std)
    # ----- Create input transformer ----
    if args.net == "resnet8":
        img_size=128
    else:
        img_size=224
    transform_augment = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15,shear=(-2,2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    transform_no_augment = transforms.Compose([
            transforms.Resize(img_size+32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    # ----- Load data -----
    write_info("Creating Train Dataset")
    trainset = places365(root=place_dir, mode='Train', transform=transform_augment)
    write_info("Creating Project Dataset")
    projectset = places365(root=place_dir, mode='Project', transform=transform_no_augment)
    write_info("Creating Test Dataset")
    testset = places365(root=place_dir, mode='Test', transform=transform_no_augment)
    _, y_num = torch.unique(torch.tensor(trainset.targets), return_counts=True)
    # print("class num list: ", y_num)
    classes = list(trainset.label_dict.keys())
    return trainset, projectset, testset, y_num.numpy(), classes


def get_covid(augment: bool, args: argparse.Namespace): 
    write_info(' ----  Dataset:  COVID ---- ')
    if args.path_covid == "":
        covid_image_dir = "data/COVID-19_Radiography_Dataset"
    else:
        covid_image_dir = args.path_covid
    nor_mean = (0.485, 0.456, 0.406)
    nor_std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=nor_mean,std=nor_std)
    # ----- Create input transformer ----
    if args.net == "resnet8":
        img_size=128
    else:
        img_size=224
    # transform_augment = transforms.Compose([
    #             transforms.Resize(img_size+32),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomCrop(img_size),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    # nor_mean = (0.485, 0.456, 0.406)
    # nor_std = (0.229, 0.224, 0.225)
    # normalize = transforms.Normalize(mean=nor_mean,std=nor_std)
    transform_augment = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15,shear=(-2,2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    transform_no_augment = transforms.Compose([
            transforms.Resize(img_size+32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    # ----- Load data -----
    write_info("Creating Train Dataset")
    trainset = COVID_Dataset(root=covid_image_dir, mode='Train', transform=transform_augment)
    write_info("Creating Project Dataset")
    projectset = COVID_Dataset(root=covid_image_dir, mode='Project', transform=transform_no_augment)
    write_info("Creating Test Dataset")
    testset = COVID_Dataset(root=covid_image_dir, mode='Test', transform=transform_no_augment)
    _, y_num = torch.unique(torch.tensor(trainset.targets), return_counts=True)
    print("class num list: ", y_num)
    classes = list(trainset.label_dict.keys())
    return trainset, projectset, testset, y_num.numpy(), classes

def get_cifar10(augment: bool, args: argparse.Namespace): 
    write_info(' ----  Dataset:  CIFAR10 ---- ')
     # ----- Create input transformer ----
    nor_mean = (0.4914, 0.4822, 0.4465)
    nor_std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=nor_mean,std=nor_std)
    if not args.net == "resnet8":
        img_size = 224
        transform_no_augment = transforms.Compose([
                                transforms.Resize(size=(img_size, img_size)),
                                transforms.ToTensor(),
                                normalize
                            ])
        if augment:
            transform = transforms.Compose([
                transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
                transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
                transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15,shear=(-2,2)),
                ]),
                transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transform_no_augment
    else:
        transform_no_augment = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform = transform_no_augment
    # ----- Load data -----
    write_info("Creating Train Dataset")
    # sys.exit()
    trainset = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform)
    write_info("Creating Project Dataset")
    projectset = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_no_augment)
    write_info("Creating Test Dataset")
    testset = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_no_augment)
    num_classes = len(np.unique(testset.targets))
    y_num = [0. for _ in range(num_classes)]
    classes = trainset.classes
    if args.long_tail:
        write_info(' ----  Dataset:  CIFAR10 in LongTail ---- ')
        # Distribute data
        list_label2indices = classify_label(trainset, num_classes)
        # heterogeneous and long_tailed setting
        _, list_label2indices_train_new = train_long_tail(deepcopy(list_label2indices), num_classes,
                                                        args.imb_factor, args.imb_type)
        all_list = []
        for l in list_label2indices_train_new:
            all_list+= l
        indices2data = Indices2Dataset(trainset)
        indices2data.load(all_list)
        indices2data.load_targets()
        for y in indices2data.targets:
            y_num[y] += 1
        return indices2data, projectset, testset, y_num, classes
        
    for y in trainset.targets:
        y_num[y] += 1
    return trainset, projectset, testset, y_num, classes


def get_cifar100(augment: bool, args: argparse.Namespace): 
    write_info(' ----  Dataset:  CIFAR100 ---- ')
     # ----- Create input transformer ----
    nor_mean = (0.4914, 0.4822, 0.4465)
    nor_std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=nor_mean,std=nor_std)
    if not args.net == "resnet8":
        img_size = 224
        transform_no_augment = transforms.Compose([
                                transforms.Resize(size=(img_size, img_size)),
                                transforms.ToTensor(),
                                normalize
                            ])
        if augment:
            transform = transforms.Compose([
                transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
                transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
                transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15,shear=(-2,2)),
                ]),
                transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transform_no_augment
    else:
        transform_no_augment = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform = transform_no_augment
    # ----- Load data -----
    write_info("Creating Train Dataset")
    # sys.exit()
    trainset = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform)
    write_info("Creating Project Dataset")
    projectset = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_no_augment)
    write_info("Creating Test Dataset")
    testset = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_no_augment)
    num_classes = len(np.unique(testset.targets))
    y_num = [0. for _ in range(num_classes)]
    classes = trainset.classes
    if args.long_tail:
        write_info(' ----  Dataset:  CIFAR100 in LongTail ---- ')
        # Distribute data
        list_label2indices = classify_label(trainset, num_classes)
        # heterogeneous and long_tailed setting
        _, list_label2indices_train_new = train_long_tail(deepcopy(list_label2indices), num_classes,
                                                        args.imb_factor, args.imb_type)
        all_list = []
        for l in list_label2indices_train_new:
            all_list+= l
        indices2data = Indices2Dataset(trainset)
        indices2data.load(all_list)
        indices2data.load_targets()
        for y in indices2data.targets:
            y_num[y] += 1
        return indices2data, projectset, testset, y_num, classes
        
    for y in trainset.targets:
        y_num[y] += 1
    return trainset, projectset, testset, y_num, classes