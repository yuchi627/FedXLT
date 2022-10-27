# From https://github.com/pliang279/LG-FedAvg/blob/7af0568b2cae88922ebeacc021b1679815092f4e/utils/train_utils.py#L24

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from copy import deepcopy
import math
import random
from itertools import permutations
import numpy as np
import torch
import pdb
from Dataset.dataset import classify_label

def fair_iid(dataset, num_users):
    """
    Sample I.I.D. client data from fairness dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fair_noniid(train_data, num_users, num_shards=200, num_imgs=300, train=True, rand_set_all=[]):
    """
    Sample non-I.I.D client data from fairness dataset
    :param dataset:
    :param num_users:
    :return:
    """
    assert num_shards % num_users == 0
    shard_per_user = int(num_shards / num_users)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    #import pdb; pdb.set_trace()

    labels = train_data[1].numpy().reshape(len(train_data[0]),)
    assert num_shards * num_imgs == len(labels)
    #import pdb; pdb.set_trace()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    if len(rand_set_all) == 0:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
            for rand in rand_set:
                rand_set_all.append(rand)

            idx_shard = list(set(idx_shard) - rand_set) # remove shards from possible choices for other users
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    else: # this only works if the train and test set have the same distribution of labels
        for i in range(num_users):
            rand_set = rand_set_all[i*shard_per_user: (i+1)*shard_per_user]
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users, rand_set_all

def iid(train_dataset, project_dataset, test_dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_classes = len(np.unique(test_dataset.targets))
    list_label2indices_train = classify_label(train_dataset, num_classes)
    list_label2indices_projection = classify_label(project_dataset, num_classes)
    list_label2indices_test = classify_label(test_dataset, num_classes)
    dict_train_users = {i:[] for i in range(num_users)}
    dict_project_users = {i:[] for i in range(num_users)}
    dict_test_users = {i:[] for i in range(num_users)}
    cls_num_list = {i:[] for i in range(num_users)}
    for class_idx in range(num_classes):
        all_train_idxs = deepcopy(list_label2indices_train[class_idx])
        all_project_idxs = deepcopy(list_label2indices_projection[class_idx])
        all_test_idxs = deepcopy(list_label2indices_test[class_idx])
        num_train_items = int(len(all_train_idxs)/num_users)
        num_project_items = int(len(all_project_idxs)/num_users)
        num_test_items = int(len(all_test_idxs)/num_users)
        for users_idx in range(num_users):
            cls_num_list[users_idx].append(num_train_items)
            dict_train_users[users_idx].extend(np.random.choice(all_train_idxs, num_train_items, replace=False))
            all_train_idxs = list(set(all_train_idxs) - set(dict_train_users[users_idx]))
            dict_project_users[users_idx].extend(np.random.choice(all_project_idxs, num_project_items, replace=False))
            all_project_idxs = list(set(all_project_idxs) - set(dict_project_users[users_idx]))
            dict_test_users[users_idx].extend(np.random.choice(all_test_idxs, num_test_items, replace=False))
            all_test_idxs = list(set(all_test_idxs) - set(dict_test_users[users_idx]))

        num_train_items = math.ceil(len(all_train_idxs)/num_users)
        num_project_items = math.ceil(len(all_project_idxs)/num_users)
        num_test_items = math.ceil(len(all_test_idxs)/num_users)
        for users_idx in range(num_users):
            cls_num_list[users_idx].append(num_train_items)
            if len(all_train_idxs) > num_train_items:
                dict_train_users[users_idx].extend(np.random.choice(all_train_idxs, num_train_items, replace=False))
                all_train_idxs = list(set(all_train_idxs) - set(dict_train_users[users_idx]))

            if len(all_project_idxs) > num_project_items:
                dict_project_users[users_idx].extend(np.random.choice(all_project_idxs, num_project_items, replace=False))
                all_project_idxs = list(set(all_project_idxs) - set(dict_project_users[users_idx]))

            if len(all_test_idxs) > num_test_items:
                dict_test_users[users_idx].extend(np.random.choice(all_test_idxs, num_test_items, replace=False))
                all_test_idxs = list(set(all_test_idxs) - set(dict_test_users[users_idx]))
    
    return dict_train_users, dict_project_users, dict_test_users, cls_num_list

# def iid(train_dataset, project_dataset, test_dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_train_items = int(len(train_dataset)/num_users)
#     dict_train_users, all_train_idxs = {}, [i for i in range(len(train_dataset))]

#     num_project_items = int(len(project_dataset)/num_users)
#     dict_project_users, all_project_idxs = {}, [i for i in range(len(project_dataset))]

#     num_test_items = int(len(test_dataset)/num_users)
#     dict_test_users, all_test_idxs = {}, [i for i in range(len(test_dataset))]

#     for users_idx in range(num_users):
#         dict_train_users[users_idx] = set(np.random.choice(all_train_idxs, num_train_items, replace=False))
#         all_train_idxs = list(set(all_train_idxs) - dict_train_users[users_idx])

#         dict_project_users[users_idx] = set(np.random.choice(all_project_idxs, num_project_items, replace=False))
#         all_project_idxs = list(set(all_project_idxs) - dict_project_users[users_idx])

#         dict_test_users[users_idx] = set(np.random.choice(all_test_idxs, num_test_items, replace=False))
#         all_test_idxs = list(set(all_test_idxs) - dict_test_users[users_idx])

#         # print(train_dataset.targets[i]  for i in dict_train_users[users_idx])
    
#     return dict_train_users, dict_project_users, dict_test_users

def get_img_num_per_cls(img_max, cls_num, imb_type, imb_factor):
    img_num_per_cls = []
    # print("img_max: ", img_max)
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / cls_num))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def imbalance_iidTrain_iidTest(train_dataset, project_dataset, test_dataset, num_users, imb_factor=0.01, imb_type='exp', rand_number=0):
    train_targets_np = np.array(train_dataset.targets, dtype=np.int64)
    project_targets_np = np.array(project_dataset.targets, dtype=np.int64)
    classes = np.unique(train_targets_np)
    class_num = len(classes)
    # np.random.shuffle(classes)

    train_class_sample_num = (len(train_targets_np)/class_num)
    train_img_num_list = get_img_num_per_cls(train_class_sample_num, class_num, imb_type, imb_factor)
    project_class_sample_num = (len(project_targets_np)/class_num)
    project_img_num_list = get_img_num_per_cls(project_class_sample_num, class_num, imb_type, imb_factor)
    # num_per_cls_dict = dict()
    # Centralize CIFAR10 Long-tailed
    dict_train_users = {}
    dict_project_users = {}
    cls_num_list = {}
    for users_idx in range(num_users):
        dict_train_users[users_idx] = list()
        dict_project_users[users_idx] = list()
        cls_num_list[users_idx] = list()
    for the_class, train_the_img_num, project_the_img_num in zip(classes, train_img_num_list, project_img_num_list):
        # num_per_cls_dict[the_class] = train_the_img_num
        print("class: ", the_class, "  img_num: ", train_the_img_num)
        train_idx = np.where(train_targets_np == the_class)[0]
        np.random.shuffle(train_idx)
        train_selec_idx = train_idx[:train_the_img_num]
        num_train_items = int(len(train_selec_idx)/num_users)
        project_idx = np.where(project_targets_np == the_class)[0]
        np.random.shuffle(project_idx)
        project_selec_idx = project_idx[:project_the_img_num]
        num_project_items = int(len(project_selec_idx)/num_users)
        # IID Split Centralize Long-tailed Train&Project Data to Client
        for users_idx in range(num_users):
            dict_train_users[users_idx] += random.choices(train_selec_idx, k=num_train_items)
            train_selec_idx = list(set(train_selec_idx) - set(dict_train_users[users_idx]))
            cls_num_list[users_idx].append(num_train_items)
            dict_project_users[users_idx] += random.choices(project_selec_idx, k=num_project_items)
            project_selec_idx = list(set(project_selec_idx) - set(dict_project_users[users_idx]))
    # IID Split IID Test Data to Client
    num_test_items = int(len(test_dataset)/num_users)
    dict_test_users, all_test_idxs = {}, [i for i in range(len(test_dataset))]
    for users_idx in range(num_users):
        dict_test_users[users_idx] = random.choices(all_test_idxs, k=num_test_items)
        all_test_idxs = list(set(all_test_idxs) - set(dict_test_users[users_idx]))
    
    return dict_train_users, dict_project_users, dict_test_users, cls_num_list

def imbalance_iidTrain_imbTest(train_dataset, project_dataset, test_dataset, num_users, imb_factor=0.01, imb_type='exp', rand_number=0):
    train_targets_np = np.array(train_dataset.targets, dtype=np.int64)
    project_targets_np = np.array(project_dataset.targets, dtype=np.int64)
    test_targets_np = np.array(test_dataset.targets, dtype=np.int64)
    classes = np.unique(train_targets_np)
    class_num = len(classes)
    # np.random.shuffle(classes)

    train_class_sample_num = (len(train_targets_np)/class_num)
    train_img_num_list = get_img_num_per_cls(train_class_sample_num, class_num, imb_type, imb_factor)
    project_class_sample_num = (len(project_targets_np)/class_num)
    project_img_num_list = get_img_num_per_cls(project_class_sample_num, class_num, imb_type, imb_factor)
    test_class_sample_num = (len(test_targets_np)/class_num)
    test_img_num_list = get_img_num_per_cls(test_class_sample_num, class_num, imb_type, imb_factor)
    # num_per_cls_dict = dict()
    # Centralize CIFAR10 Long-tailed
    dict_train_users = {}
    dict_project_users = {}
    dict_test_users = {}
    for users_idx in range(num_users):
        dict_train_users[users_idx] = list()
        dict_project_users[users_idx] = list()
        dict_test_users[users_idx] = list()
    for the_class, train_the_img_num, project_the_img_num, test_the_img_num in zip(classes, train_img_num_list, project_img_num_list, test_img_num_list):
        # num_per_cls_dict[the_class] = train_the_img_num
        train_idx = np.where(train_targets_np == the_class)[0]
        np.random.shuffle(train_idx)
        train_selec_idx = train_idx[:train_the_img_num]
        num_train_items = int(len(train_selec_idx)/num_users)
        project_idx = np.where(project_targets_np == the_class)[0]
        np.random.shuffle(project_idx)
        project_selec_idx = project_idx[:project_the_img_num]
        num_project_items = int(len(project_selec_idx)/num_users)
        test_idx = np.where(test_targets_np == the_class)[0]
        np.random.shuffle(test_idx)
        test_selec_idx = test_idx[:test_the_img_num]
        num_test_items = int(len(test_selec_idx)/num_users)
        # IID Split Centralize Long-tailed Train&Project&Test Data to Client
        for users_idx in range(num_users):
            dict_train_users[users_idx] += random.choices(train_selec_idx, k=num_train_items)
            train_selec_idx = list(set(train_selec_idx) - set(dict_train_users[users_idx]))
            dict_project_users[users_idx] += random.choices(project_selec_idx, k=num_project_items)
            project_selec_idx = list(set(project_selec_idx) - set(dict_project_users[users_idx]))
            dict_test_users[users_idx] += random.choices(test_selec_idx, k=num_test_items)
            test_selec_idx = list(set(test_selec_idx) - set(dict_test_users[users_idx]))
            
            
    return dict_train_users, dict_project_users, dict_test_users

def imbalance_exp_iidTest(train_dataset, project_dataset, test_dataset, num_users, imb_factor=0.01, imb_type='exp', rand_number=0):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    train_idx_per_cls_dict = {}
    # test_idx_per_cls_dict = {}
    project_idx_per_cls_dict = {}
    dict_train_users = {}
    # dict_test_users = {}
    dict_project_users = {}
    train_targets_np = np.array(train_dataset.targets, dtype=np.int64)
    # test_targets_np = np.array(test_dataset.targets, dtype=np.int64)
    project_targets_np = np.array(project_dataset.targets, dtype=np.int64)
    classes = np.unique(train_targets_np)
    class_num = len(classes)
    train_class_sample_num = (len(train_targets_np)/class_num)/num_users
    # test_class_sample_num = (len(test_targets_np)/class_num)/num_users
    project_class_sample_num = (len(project_targets_np)/class_num)/num_users

    num_test_items = int(len(test_dataset)/num_users)
    dict_test_users, all_test_idxs = {}, [i for i in range(len(test_dataset))]

    # ----- create dictionary of all class label index ------
    for the_class in classes:
        train_idx_per_cls_dict[the_class] = list(np.where(train_targets_np == the_class)[0])
        # test_idx_per_cls_dict[the_class] = list(np.where(test_targets_np == the_class)[0])
        project_idx_per_cls_dict[the_class] = list(np.where(project_targets_np == the_class)[0])


    for users_idx in range(num_users):
        # print("================= User: ", users_idx)
        # -------- random number of class and random classes -------- 
        choice_imbalance_class_num = random.randint(1, class_num - 1)
        imbalance_class_idx = random.sample(range(class_num),choice_imbalance_class_num)
        # -------- define number of samples of each class (follow exp) -------- 
        img_num_per_cls_train = get_img_num_per_cls(train_class_sample_num, choice_imbalance_class_num, imb_type, imb_factor)
        # img_num_per_cls__test = get_img_num_per_cls(test_class_sample_num, choice_imbalance_class_num, imb_type, imb_factor)
        img_num_per_cls_project = get_img_num_per_cls(project_class_sample_num, choice_imbalance_class_num, imb_type, imb_factor)
        imbalanced_train_idx = []
        # imbalanced_test_idx = []
        imbalanced_project_idx = []
        # print("Sample class: ", imbalance_class_idx)
        # print("train class sample number: ", img_num_per_cls_train)
        for n, the_class in enumerate(imbalance_class_idx):
            # print("Class: ", the_class)
            # print("ALL sample number of train class: ", len(train_idx_per_cls_dict[the_class]))
            # print("Gonna sample number: ", img_num_per_cls_train[n])
            train_sample = random.sample(train_idx_per_cls_dict[the_class], img_num_per_cls_train[n])
            imbalanced_train_idx.extend(train_sample)
            train_idx_per_cls_dict[the_class] = list(set(train_idx_per_cls_dict[the_class]) - set(train_sample))
            # print("Remain sample number of train class: ", len(train_idx_per_cls_dict[the_class]))
            
            
            # print("ALL sample number of test class: ", len(test_idx_per_cls_dict[the_class]))
            # print("Gonna sample number: ", img_num_per_cls__test[n])
            # test_sample = random.sample(test_idx_per_cls_dict[the_class], img_num_per_cls__test[n])
            # imbalanced_test_idx.extend(test_sample)
            # test_idx_per_cls_dict[the_class] = list(set(test_idx_per_cls_dict[the_class]) - set(test_sample))
            # print("Remain sample number of test class: ", len(test_idx_per_cls_dict[the_class]))
            
            
            # print("ALL sample number of project class: ", len(project_idx_per_cls_dict[the_class]))
            # print("Gonna sample number: ", img_num_per_cls_project[n])
            project_sample = random.sample(project_idx_per_cls_dict[the_class], img_num_per_cls_project[n])
            imbalanced_project_idx.extend(project_sample)
            project_idx_per_cls_dict[the_class] = list(set(project_idx_per_cls_dict[the_class]) - set(project_sample))
            # print("Remain sample number of project class: ", len(project_idx_per_cls_dict[the_class]))

        dict_test_users[users_idx] = set(np.random.choice(all_test_idxs, num_test_items, replace=False))
        all_test_idxs = list(set(all_test_idxs) - dict_test_users[users_idx])

        dict_train_users[users_idx] = imbalanced_train_idx
        # dict_test_users[users_idx] = imbalanced_test_idx
        dict_project_users[users_idx] = imbalanced_project_idx
    return dict_train_users, dict_project_users, dict_test_users



def imbalance_exp(train_dataset, project_dataset, test_dataset, num_users, imb_factor=0.01, imb_type='exp', rand_number=0):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    train_idx_per_cls_dict = {}
    test_idx_per_cls_dict = {}
    project_idx_per_cls_dict = {}
    dict_train_users = {}
    dict_test_users = {}
    dict_project_users = {}
    train_targets_np = np.array(train_dataset.targets, dtype=np.int64)
    test_targets_np = np.array(test_dataset.targets, dtype=np.int64)
    project_targets_np = np.array(project_dataset.targets, dtype=np.int64)
    classes = np.unique(train_targets_np)
    class_num = len(classes)
    train_class_sample_num = (len(train_targets_np)/class_num)/num_users
    test_class_sample_num = (len(test_targets_np)/class_num)/num_users
    project_class_sample_num = (len(project_targets_np)/class_num)/num_users

    # ----- create dictionary of all class label index ------
    for the_class in classes:
        train_idx_per_cls_dict[the_class] = list(np.where(train_targets_np == the_class)[0])
        test_idx_per_cls_dict[the_class] = list(np.where(test_targets_np == the_class)[0])
        project_idx_per_cls_dict[the_class] = list(np.where(project_targets_np == the_class)[0])

    for users_idx in range(num_users):
        # print("================= User: ", users_idx)
        # -------- random number of class and random classes -------- 
        choice_imbalance_class_num = random.randint(1, class_num - 1)
        imbalance_class_idx = random.sample(range(class_num),choice_imbalance_class_num)
        # -------- define number of samples of each class (follow exp) -------- 
        img_num_per_cls_train = get_img_num_per_cls(train_class_sample_num, choice_imbalance_class_num, imb_type, imb_factor)
        img_num_per_cls__test = get_img_num_per_cls(test_class_sample_num, choice_imbalance_class_num, imb_type, imb_factor)
        img_num_per_cls_project = get_img_num_per_cls(project_class_sample_num, choice_imbalance_class_num, imb_type, imb_factor)
        imbalanced_train_idx = []
        imbalanced_test_idx = []
        imbalanced_project_idx = []
        # print("Sample class: ", imbalance_class_idx)
        # print("train class sample number: ", img_num_per_cls_train)
        for n, the_class in enumerate(imbalance_class_idx):
            # print("Class: ", the_class)
            # print("ALL sample number of train class: ", len(train_idx_per_cls_dict[the_class]))
            # print("Gonna sample number: ", img_num_per_cls_train[n])
            train_sample = random.sample(train_idx_per_cls_dict[the_class], img_num_per_cls_train[n])
            imbalanced_train_idx.extend(train_sample)
            train_idx_per_cls_dict[the_class] = list(set(train_idx_per_cls_dict[the_class]) - set(train_sample))
            # print("Remain sample number of train class: ", len(train_idx_per_cls_dict[the_class]))
            
            
            # print("ALL sample number of test class: ", len(test_idx_per_cls_dict[the_class]))
            # print("Gonna sample number: ", img_num_per_cls__test[n])
            test_sample = random.sample(test_idx_per_cls_dict[the_class], img_num_per_cls__test[n])
            imbalanced_test_idx.extend(test_sample)
            test_idx_per_cls_dict[the_class] = list(set(test_idx_per_cls_dict[the_class]) - set(test_sample))
            # print("Remain sample number of test class: ", len(test_idx_per_cls_dict[the_class]))
            
            
            # print("ALL sample number of project class: ", len(project_idx_per_cls_dict[the_class]))
            # print("Gonna sample number: ", img_num_per_cls_project[n])
            project_sample = random.sample(project_idx_per_cls_dict[the_class], img_num_per_cls_project[n])
            imbalanced_project_idx.extend(project_sample)
            project_idx_per_cls_dict[the_class] = list(set(project_idx_per_cls_dict[the_class]) - set(project_sample))
            # print("Remain sample number of project class: ", len(project_idx_per_cls_dict[the_class]))

        dict_train_users[users_idx] = imbalanced_train_idx
        dict_test_users[users_idx] = imbalanced_test_idx
        dict_project_users[users_idx] = imbalanced_project_idx
    return dict_train_users, dict_project_users, dict_test_users


# def iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all

def noniid_replace(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users, rand_set_all

