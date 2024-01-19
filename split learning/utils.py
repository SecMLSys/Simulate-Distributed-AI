import os
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np
import random, math
import gmm

from split_models.encoder import *
from split_models.decoder import *
from split_models.classifier import *

import sys
sys.path.insert(1, '../')

from data_preprocess.load_health import load_health
from data_preprocess.load_utkface import load_utkface
from data_preprocess.load_nlps import load_twitter, getvocab
from data_preprocess.load_german import load_german
from data_preprocess.load_mnist import load_mnist
from data_preprocess.load_purchase100 import load_purchase, load_purchase_reference_data
from data_preprocess.load_texas100 import load_texas, load_texas_reference_data
from data_preprocess.load_cifar100 import load_cifar100, load_cifar100_reference_data
from data_preprocess.load_cifar10 import load_cifar10, load_cifar10_reference_data

from data_preprocess.config import dataset_class_params

from torch.nn.utils.rnn import pad_sequence as pad_sequence


def set_random_seed(args):
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg = recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def load_data(args):
    if args.dataset == 'german':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_german(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)
    if args.dataset == 'health':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_health(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        binarize=False,
                                                                                                        batch_size=args.batch_size)
    elif args.dataset == 'utkface':
        train_loader, test_loader, img_dim, target_num_classes, sensitive_num_classes = load_utkface(target_attr=args.target_attr,
                                                                                                        sensitive_attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)
    elif args.dataset == 'twitter':
        train_loader, test_loader, voc_size, target_num_classes, sensitive_num_classes = load_twitter(sensitive_attr=args.sensitive_attr,
                                                                                                      random_seed=args.seed,
                                                                                                      batch_size=args.batch_size)

    elif args.dataset == 'mnist':
        train_loader, test_loader = load_mnist(args)

    elif args.dataset == 'purchase':
        train_loader, test_loader = load_purchase(batch_size=args.batch_size)

    elif args.dataset == 'texas':
        train_loader, test_loader = load_texas(batch_size=args.batch_size)

    elif args.dataset == 'cifar100':
        train_loader, test_loader = load_cifar100(batch_size=args.batch_size)

    elif args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(batch_size=args.batch_size)

    return train_loader, test_loader


def partition_train_data(train_loader, args, train_size=0.9):

    dataset = train_loader.dataset

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(train_size*len(dataset)),
                                                    len(dataset)-int(train_size*len(dataset))],
                                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    return train_loader, val_loader


def partition_test_data(test_loader, args, val_size=0.5):

    dataset = test_loader.dataset

    val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(val_size*len(dataset)),
                                                    len(dataset)-int(val_size*len(dataset))],
                                                    generator=torch.Generator().manual_seed(10000))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=2,
                                   drop_last=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    return test_loader, val_loader


def twitter_collate_batch_gender(batch):
    text_list, y_list, s_list = [], [], []

    for (text, y, _, s) in batch:
         text = torch.tensor(text, dtype=torch.int64)
         text_list.append(text)
         y_list.append(y)
         s_list.append(s)

    y_list = torch.tensor(y_list, dtype=torch.int64)
    s_list = torch.tensor(s_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=1)

    return text_list, y_list, s_list



def partition_test_twitter_data(test_loader, args, val_size=0.5):

    dataset = test_loader.dataset

    val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(val_size*len(dataset)),
                                                    len(dataset)-int(val_size*len(dataset))],
                                                    generator=torch.Generator().manual_seed(10000))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=2,
                                   collate_fn=twitter_collate_batch_gender,
                                   drop_last=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=2,
                                   collate_fn=twitter_collate_batch_gender,
                                   drop_last=True)

    return test_loader, val_loader




def load_reference_data(args):
    if args.dataset == 'purchase':
        reference_loader = load_purchase_reference_data(batch_size=args.batch_size)
    elif args.dataset == 'texas':
        reference_loader = load_texas_reference_data(batch_size=args.batch_size)
    elif args.dataset == 'cifar100':
        reference_loader = load_cifar100_reference_data(batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        reference_loader = load_cifar10_reference_data(batch_size=args.batch_size)

    return reference_loader


def load_encoder(args, info_model=False):
    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        local_encoder = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        local_encoder = health_encoder(input_dim=input_dim)

    elif args.dataset == 'utkface':
        # img_dim = dataset_class_params['utkface']['img_dim']
        try:
            if args.model_architecture == 'vgg':
                local_encoder = vgg_encoder()
            elif args.model_architecture == 'resnet':
                local_encoder = resnet_encoder()
        except:
            print(info_model)
            local_encoder = vgg_encoder()

    elif args.dataset == 'twitter':
        voc_size = dataset_class_params['twitter']['voc_size']
        local_encoder = lstm_encoder(voc_size=voc_size)

    elif args.dataset == 'mnist':
        if args.model_architecture == 'vgg':
            local_encoder = vgg_encoder(img_channels=1)
        elif args.model_architecture == 'resnet':
            local_encoder = resnet_encoder(img_channels=1)

    elif args.dataset == 'purchase':
        local_encoder = purchase_encoder()

    elif args.dataset == 'texas':
        local_encoder = texas_encoder()

    elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
        local_encoder = vgg_encoder()

    return local_encoder



def load_decoder(args):
    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        decoder = mlp_decoder(in_dim=args.num_features,
                              out_dim=input_dim,
                              drop_rate=args.drop_rate)


    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        decoder = mlp_decoder(in_dim=args.num_features,
                              out_dim=input_dim,
                              drop_rate=args.drop_rate)


    elif args.dataset == 'utkface' or args.dataset == 'mnist':
        # img_dim = dataset_class_params['utkface']['img_dim']
        decoder = vgg_decoder()

    elif args.dataset == 'twitter':
        voc_size = dataset_class_params['twitter']['voc_size']
        decoder = lstm_decoder(voc_size=voc_size)

    elif args.dataset == 'mnist':
        if args.model_architecture == 'vgg':
            local_encoder = vgg_decoder(img_dim=28, img_channels=1)
        elif args.model_architecture == 'resnet':
            local_encoder = resnet_decoder(img_dim=28, img_channels=1)

    return decoder


def load_classifier(args, num_classes=None):

    if num_classes is None:
        num_classes = dataset_class_params[args.dataset][args.target_attr]

    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        classifier = health_classifier(num_classes=num_classes)


    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        classifier = health_classifier(num_classes=num_classes)


    elif args.dataset == 'utkface' or args.dataset == 'mnist':
        img_dim = dataset_class_params[args.dataset]['img_dim']
        try:
            if args.model_architecture == 'vgg':
                classifier = vgg_classifier(img_dim=img_dim, num_classes=num_classes)
            elif args.model_architecture == 'resnet':
                classifier = resnet_classifier(num_classes=num_classes)
        except:
            classifier = vgg_classifier(img_dim=img_dim, num_classes=num_classes)

        return classifier

    elif args.dataset == 'twitter':
        # voc_size = dataset_class_params['twitter']['voc_size']
        classifier = lstm_classifier(num_classes=num_classes)

    elif args.dataset == 'purchase':
        classifier = purchase_classifier()

    elif args.dataset == 'texas':
        classifier = texas_classifier()

    elif args.dataset == 'cifar100':
        classifier = vgg_classifier(img_dim=32, num_classes=100)

    elif args.dataset == 'cifar10':
        classifier = vgg_classifier(img_dim=32, num_classes=10)

    return classifier


def load_privacy_examiner(args, num_classes=None):

    if num_classes is None:
        num_classes = dataset_class_params[args.dataset][args.sensitive_attr]

    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        classifier = health_classifier(num_classes=num_classes)


    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        classifier = health_classifier(num_classes=num_classes)


    elif args.dataset == 'utkface' or args.dataset == 'mnist':
        img_dim = dataset_class_params[args.dataset]['img_dim']
        try:
            if args.model_architecture == 'vgg':
                classifier = vgg_classifier(img_dim=img_dim, num_classes=num_classes)
            elif args.model_architecture == 'resnet':
                classifier = resnet_classifier(num_classes=num_classes)
        except:
            classifier = vgg_classifier(img_dim=img_dim, num_classes=num_classes)

        return classifier

    elif args.dataset == 'twitter':
        # voc_size = dataset_class_params['twitter']['voc_size']
        classifier = lstm_classifier(num_classes=num_classes)

    elif args.dataset == 'purchase':
        classifier = purchase_classifier()

    elif args.dataset == 'texas':
        classifier = texas_classifier()

    elif args.dataset == 'cifar100':
        classifier = vgg_classifier(img_dim=32, num_classes=100)

    elif args.dataset == 'cifar10':
        classifier = vgg_classifier(img_dim=32, num_classes=10)

    return classifier


def normal_weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.01)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.01)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.01)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def numpy_to_textlist(data):
    voc_size = dataset_class_params['twitter']['voc_size']

    vocabulary = getvocab()
    textlist = []

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    for idx in range(data.shape[1]):
        textidxlist = data[:, idx].tolist()
        textlist.append([vocabulary.itos[int(textidxlist[i])] for i in range(len(textidxlist))])

    return textlist


def model_memory(models):
    mem = 0
    for model in models:
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem += mem_params + mem_bufs
    return mem
