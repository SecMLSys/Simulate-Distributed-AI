
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

def load_texas(batch_size=128, random_seed=0):
    DATASET_PATH = './texas/'

    DATASET_FEATURES = os.path.join(DATASET_PATH,'texas/100/feats')
    DATASET_LABELS = os.path.join(DATASET_PATH,'texas/100/labels')
    DATASET_NUMPY = 'data.npz'

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    if not os.path.isfile(DATASET_FEATURES):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')

        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)
        print('reading dataset...')
        data_set_features = np.genfromtxt(DATASET_FEATURES,delimiter=',')
        data_set_label = np.genfromtxt(DATASET_LABELS,delimiter=',')
        print('finish reading!')

        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)


    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    data_len = len(X)

    np.random.seed(random_seed)
    r = np.arange(data_len)
    np.random.shuffle(r)
    print(r[0:2])
    # r = np.load('../data_preprocess/dataset_shuffle/random_r_texas100.npy')
    X, Y = X[r], Y[r]

    num_train = 20000
    # refer_ratio = 0.3
    #
    # train_data = X[:num_train]
    # test_data = X[int(num_train + refer_ratio*data_len):]
    #
    # train_label = Y[:num_train]
    # test_label = Y[int(num_train + refer_ratio*data_len):]

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


def load_texas_reference_data(batch_size=128, random_seed=0):
    DATASET_PATH = './texas/'

    DATASET_FEATURES = os.path.join(DATASET_PATH,'texas/100/feats')
    DATASET_LABELS = os.path.join(DATASET_PATH,'texas/100/labels')
    DATASET_NUMPY = 'data.npz'

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    if not os.path.isfile(DATASET_FEATURES):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')

        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)
        print('reading dataset...')
        data_set_features = np.genfromtxt(DATASET_FEATURES,delimiter=',')
        data_set_label = np.genfromtxt(DATASET_LABELS,delimiter=',')
        print('finish reading!')

        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)-1
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
