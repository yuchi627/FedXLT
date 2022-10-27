import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import copy
import os
from PIL import Image
from torchvision import transforms
import random
import json

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset.targets):
        list1[datum].append(idx)
    return list1

def show_clients_data_distribution(dataset, clients_indices: list, num_classes, writer):
    dict_per_client = []
    cls_num = {}
    lables = dataset.targets
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = lables[idx]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        for cls, num in enumerate(nums_data):
            if client == 0:
                cls_num[cls] = {}
            cls_num[cls][str(client)] = num
        # print(f'{client}: {nums_data}')
    if not writer is None:
        for cls in cls_num:
            writer.add_scalars('Distribution/Each_clients_data', cls_num[cls], cls)
    return dict_per_client

def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach

def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel

def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.cpu().detach().float()
        self.targets = labels.cpu().detach()
    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def __len__(self):
        return self.images.shape[0]

class ServerDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.cpu().detach().numpy()
        self.targets = labels.cpu().detach().numpy()
        
    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def __len__(self):
        return self.images.shape[0]


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        self.targets = None
    def load(self, indices: list):
        self.indices = indices
    
    def load_targets(self):
        self.targets = [self.dataset.targets[i] for i in self.indices]

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)

class COVID_Dataset(Dataset):
    def __init__(self, root="../../sda2/prototree-fl/data/COVID-19_Radiography_Dataset", mode='Train', transform=None, target_transform=None):
        self.label_dict={"Normal":0, "Lung_Opacity": 1, "COVID": 2, "Viral Pneumonia": 3}
        self.img_path = []
        self.labels = []
        if mode == 'Train':
            txt = os.path.join(root, "train.txt")
        elif mode == 'Test':
            txt = os.path.join(root, "test.txt")
        else:
            txt = os.path.join(root, "train.txt")
        with open(txt, 'r') as f:
            for line in f:
                split = line.split()
                label = split[-1]
                path = split[0]
                for p in split[1:-1]:
                    path += " "+p
                self.img_path.append(os.path.join(root, path))
                self.labels.append(int(label))
        self.targets = self.labels  # Sampler needs to use targets
        self.transform = transform
        if transform == None:
            self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                    ])
            if mode != 'Train':
                self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                    ])
        else:
            self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.labels)

   
class places365(Dataset):
    def __init__(self, root, mode='Train', transform=None):
        self.label_dict=json.load(open(os.path.join(root, "dict_label.txt")))
        self.img_path = []
        self.labels = []
        self.transform = transform
        if mode == 'Test':
            txt = os.path.join(root, "Places_LT_test.txt")
        else:
            txt = os.path.join(root, "Places_LT_train.txt")
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label  # , index