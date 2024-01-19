
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

def load_cifar10(batch_size=128, random_seed=0):

    train_dataset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transforms.ToTensor())

    train_dataset, reference_dataset, __ = torch.utils.data.random_split(train_dataset, 
                [20000, 20000, 10000], generator=torch.Generator().manual_seed(0))

    test_dataset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print('Data loading finished')
    return train_loader, test_loader


def load_cifar10_reference_data(batch_size=128, random_seed=0):
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transforms.ToTensor())

    train_dataset, reference_dataset, __ = torch.utils.data.random_split(train_dataset, 
                [20000, 20000, 10000], generator=torch.Generator().manual_seed(0))

    reference_loader = torch.utils.data.DataLoader(reference_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print('Reference Data loading finished')
    return reference_loader
