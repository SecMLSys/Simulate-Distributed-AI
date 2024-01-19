
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import tarfile

def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

def load_purchase(batch_size=128, random_seed=0):
    DATASET_PATH='./purchase'
    DATASET_NAME= 'dataset_purchase'
    DATASET_NUMPY = 'data.npz'

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

    if not os.path.isfile(DATASET_FILE):
        print('Dowloading the dataset...')
        filename = "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"
        urllib.request.urlretrieve(filename, os.path.join(DATASET_PATH, 'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

        print('reading dataset...')
        data_set =np.genfromtxt(DATASET_FILE, delimiter=',')
        print('finish reading!')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)

    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    data_len = len(X)
    np.random.seed(random_seed)
    r = np.arange(data_len)
    np.random.shuffle(r)
    print(r[0:2])
    X, Y = X[r], Y[r]

    num_train = 20000

    train_data = X[:num_train]
    test_data = X[num_train:num_train*2]

    train_label = Y[:num_train]
    test_label = Y[num_train:num_train*2]

    train_data = tensor_data_create(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)

    test_data = tensor_data_create(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    print('Data loading finished')
    return train_loader, test_loader


def load_purchase_reference_data(batch_size=128, random_seed=0):
    DATASET_PATH='./purchase'
    DATASET_NAME= 'dataset_purchase'
    DATASET_NUMPY = 'data.npz'

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

    if not os.path.isfile(DATASET_FILE):
        print('Dowloading the dataset...')
        filename = "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"
        urllib.request.urlretrieve(filename, os.path.join(DATASET_PATH, 'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

        print('reading dataset...')
        data_set =np.genfromtxt(DATASET_FILE, delimiter=',')
        print('finish reading!')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)

    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    data_len = len(X)
    np.random.seed(random_seed)
    r = np.arange(data_len)
    np.random.shuffle(r)
    print(r[0:2])
    X, Y = X[r], Y[r]

    num_train = 20000

    reference_data = X[num_train*2:num_train*3]

    reference_label = Y[num_train*2:num_train*3]

    reference_data = tensor_data_create(reference_data, reference_label)
    reference_loader = torch.utils.data.DataLoader(reference_data, batch_size=batch_size, shuffle=True, num_workers=1)

    print('Reference Data loading finished')
    return reference_loader
