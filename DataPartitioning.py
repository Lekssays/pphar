import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True


def partition_cifar():
    num_clients = 20
    num_selected = 6
    num_rounds = 150
    epochs = 5
    batch_size = 32

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Loading CIFAR10 using torchvision.datasets
    traindata = datasets.CIFAR10('./data', train=True, download=True,
                        transform= transform_train)

    # Dividing the training data into num_clients, with each client having equal number of images
    # traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
    list_len = [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)]
    traindata_split = torch.utils.data.random_split(traindata,list_len)

    # Creating a pytorch loader for a Deep Learning model
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]



    # Normalizing the test images
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Loading the test iamges and thus converting them into a test_loader
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            ), batch_size=batch_size, shuffle=True)

    i = 1
    path = "/home/deba/workspace/pphar/data/cifar10/"
    for train_data in train_loader:
        
        if os.path.exists(path + str(i)+"/"):
            print("Folder already exists")
        else:
            os.makedirs(path + str(i)+"/")
        train_data_path = path + str(i)+"/"+"train_data_ind.pth"
        test_data_path = path + str(i)+"/"+"test_data_ind.pth"
        torch.save(train_data, train_data_path)
        torch.save(test_loader, test_data_path)
        i+=1

def partition_fmist():

    num_clients = 20
    num_selected = 6
    num_rounds = 150
    epochs = 5
    batch_size = 32

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # Loading CIFAR10 using torchvision.datasets
    traindata = datasets.FashionMNIST('./data', train=True, download=True,
                        transform= transform_train)

    # Dividing the training data into num_clients, with each client having equal number of images
    traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])

    # Creating a pytorch loader for a Deep Learning model
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]



    # Normalizing the test images
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # Loading the test iamges and thus converting them into a test_loader
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform_test
            ), batch_size=batch_size, shuffle=True)

    i = 1
    path = "/home/deba/workspace/pphar/data/fmnist/"
    for train_data in train_loader:
        
        if os.path.exists(path + str(i)+"/"):
            print("Folder already exists")
        else:
            os.makedirs(path + str(i)+"/")
        train_data_path = path + str(i)+"/"+"train_data_ind.pth"
        test_data_path = path + str(i)+"/"+"test_data_ind.pth"
        torch.save(train_data, train_data_path)
        torch.save(test_loader, test_data_path)
        i+=1

if __name__ == "__main__":
    partition_cifar()
    # partition_fmist()