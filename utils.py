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

    print('correct')

    return test_loader, val_loader

def data_generator(data_loader, sensitive=True):
    while True:
        for _, data in enumerate(data_loader):
            yield data


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




## functions for studying privacy and fairness

def one_hot_encoding(label, num_classes, use_cuda=False):
    label_cpu = label.cpu().data
    onehot = torch.zeros(label_cpu.size(0), num_classes).scatter_(1, label_cpu.unsqueeze(1), 1.).float()
    return onehot.cuda() if use_cuda else onehot

def pairwise_kl(mu, sigma, add_third_term=False):
    # k = K.shape(sigma)[1]
    # k = K.cast(k, 'float32')
    d = float(sigma.size(1))

    # var = K.square(sigma) + 1e-8
    var = sigma**2 + 1e-8

    # var_inv = K.tf.reciprocal(var)
    # var_diff = K.dot(var, K.transpose(var_inv))
    var_inv = 1./var
    var_diff = torch.matmul(var, torch.t(var_inv)) ## tr(S2^-1 S1)

    # r = K.dot(mu * mu, K.transpose(var_inv))
    # r2 = K.sum(mu * mu * var_inv, axis=1)
    # mu_var_mu = 2 * K.dot(mu, K.transpose(mu * var_inv))
    # mu_var_mu = r - mu_var_mu + K.transpose(r2)

    r = torch.matmul(mu**2, torch.t(var_inv)) ## batch x batch
    r2 = torch.sum(mu*mu*var_inv, dim=1, keepdim=True) ## batch x 1
    mu_var_mu = 2 * torch.matmul(mu, torch.t(mu * var_inv))
    mu_var_mu = r - mu_var_mu + torch.t(r2)

    if add_third_term:  # this term cancels out for examples in a batch (meaning = 0)
        log_det = torch.sum(torch.log(var), dim=1, keepdim=True)
        log_def_diff = log_det - torch.t(log_det)
    else:
        log_det_diff = 0.

    KL = 0.5 * (var_diff + mu_var_mu + log_det_diff - d) ## batch x batch
    return KL



def kl_conditional_and_marg(mu, sigma):
    b = float(sigma.size(0))
    d = float(sigma.size(1))

    ### H(z|x)
    H = 0.5*(torch.sum(torch.log(sigma**2 + 1e-8), dim=1)
                                        + d*(1 + math.log(2 * math.pi))) ## d/2*log(2e\pi det)

    KL = pairwise_kl(mu, sigma)


    return 1.0/b * torch.mean(torch.sum(KL, dim=1) + (b - 1) * H - math.log(b))

def information_bottleneck(mu, sigma):
    mu = mu.contiguous()
    mu = mu.view(mu.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)
    return F.relu(torch.mean(pairwise_kl(mu, sigma))) ## to guarantee the estimation is larger than 0

def conditional_gaussian_entropy(mu, sigma):
    d = float(sigma.size(1))
    H = 0.5*(torch.sum(torch.log(sigma**2 + 1e-8), dim=1)
                                         + d*(1 + math.log(2 * math.pi)))
    return torch.mean(H)




##--------------mutual information estimations--------------------


def label_entropy(target_num_classes, labels):
    label_ent = 0
    for target in range(target_num_classes):
        label_probability = (labels == target).sum().item()/float(labels.size(0))
        if label_probability > 0:
            label_ent -= label_probability * np.log(label_probability)
    return label_ent

def representation_data_mutual_information(mu, sigma):
    mu = mu.contiguous()
    mu = mu.view(mu.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)

    KL = F.relu(pairwise_kl(mu, sigma))
    KL_exp = torch.exp(-KL)

    return torch.mean(-torch.log(torch.mean(KL_exp, dim=1)+1e-8))


def label_sensitive_mutual_information(label, s, target_num_classes, sensitive_num_classes):

    mutual_info = 0
    label_cpu = label.cpu()
    one_hot_label = torch.zeros(label_cpu.size(0),
            target_num_classes).scatter_(1, label_cpu.unsqueeze(1), 1.).float().to(s.device)

    class_prob = torch.sum(one_hot_label, dim=0)/torch.sum(one_hot_label)

    counter = 0
    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        slabel_pred = slabel_mask.unsqueeze(1) * one_hot_label

        if torch.sum(slabel_pred) > 0:
            cond_prob = torch.sum(slabel_pred, dim=0)/(torch.sum(slabel_pred))
            mutual_info += torch.sum(cond_prob*torch.log(cond_prob/(class_prob + 1e-8) + 1e-8))
            counter += 1

    return mutual_info/float(counter)

def label_sensitive_mutual_information_v2(trainloader, target_num_classes, sensitive_num_classes):
    mutual_info = 0
    class_prob = torch.tensor([0.0 for i in range(target_num_classes)])
    cond_prob = [torch.tensor([0.0 for i in range(target_num_classes)]) for i in range(sensitive_num_classes)]
    num_samples = torch.tensor(0.0)
    cond_num_samples = [torch.tensor(0.0) for i in range(sensitive_num_classes)]
    for i, (_, y, s) in enumerate(trainloader):
        label_cpu = y.cpu()
        one_hot_label = torch.zeros(label_cpu.size(0),
                target_num_classes).scatter_(1, label_cpu.unsqueeze(1), 1.).float().to(s.device)

        class_prob += torch.sum(one_hot_label, dim=0)
        num_samples += torch.sum(one_hot_label)

        for slabel in range(sensitive_num_classes):
            slabel_mask = torch.eq(s, slabel).float()
            slabel_pred = slabel_mask.unsqueeze(1) * one_hot_label

            if torch.sum(slabel_pred) > 0:
                cond_prob[slabel] += torch.sum(slabel_pred, dim=0)
                cond_num_samples[slabel] += torch.sum(slabel_pred)

    class_prob = class_prob/num_samples
    print(class_prob)

    for slabel in range(sensitive_num_classes):
        cond_prob[slabel] = cond_prob[slabel]/cond_num_samples[slabel]

        mutual_info += torch.sum(cond_prob[slabel]*torch.log(cond_prob[slabel]/(class_prob)))

    print(mutual_info)

    return mutual_info/float(sensitive_num_classes)

def representation_sensitive_mutual_information_v2(mu, sigma, s, sensitive_num_classes):
    mu = mu.contiguous()
    mu = mu.view(mu.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)
    KL = F.relu(pairwise_kl(mu, sigma))
    # print(KL)
    KL_exp = torch.exp(-KL)
    # print(KL_exp)

    mutual_info = 0
    numerator = torch.mean(KL_exp, dim=1)
    # print(numerator)

    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        total_slabel = torch.sum(slabel_mask)
        denominator = torch.sum(slabel_mask.repeat(KL_exp.size(0), 1)*KL_exp, dim=1)/(total_slabel+1e-8) + 1e-8
        mutual_info += torch.mean(torch.log(numerator/denominator))

    return mutual_info/float(sensitive_num_classes)

def representation_sensitive_mutual_information(mu, sigma, s, sensitive_num_classes, iters=10):
    # print(mu.size())
    mu = mu.contiguous()
    mu = mu.view(mu.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)

    KL = F.relu(pairwise_kl(mu, sigma))
    KL_exp = torch.exp(-KL)
    N = float(s.size(0))

    mutual_info = 0

    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        total_slabel = torch.sum(slabel_mask)

        if total_slabel > 0:
            slabel_mask_matrix = slabel_mask.repeat(KL_exp.size(0), 1)
            phi = slabel_mask_matrix/N/float(total_slabel)
            
            for iter in range(iters):
                psi = 1./N*phi.t()/(torch.sum(phi, dim=1) + 1e-8)
                inter_mat = psi*slabel_mask_matrix.t()*KL_exp
                phi = (1./float(total_slabel)*inter_mat.t())/(torch.sum(inter_mat, dim=1) + 1e-8)

            phi, psi = phi.detach(), psi.detach()

            mutual_info += torch.sum(phi.t()*slabel_mask_matrix.t()*KL)

    return mutual_info/float(sensitive_num_classes)


def membership_mutual_information(mu, sigma, mu_val, sigma_val, iters=10):
    mu = mu.contiguous()
    mu = mu.view(mu.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)
    mu_val = mu_val.view(mu_val.size(0), -1)
    sigma_val = sigma_val.view(sigma_val.size(0), -1)

    m = torch.cat([torch.zeros(mu.size(0)), torch.ones(mu_val.size(0))]).to(mu.device)
    mu_cat = torch.cat([mu, mu_val], dim=0)
    sigma_cat = torch.cat([sigma, sigma_val], dim=0)
    mutual_info = representation_sensitive_mutual_information_v2(mu_cat, sigma_cat, m, 2)
    return mutual_info


def mcmc_mutual_information_estimator(mu, sigma, s, sensitive_num_classes, n=50000):

    mu = mu.view(mu.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)

    sigma = np.square(sigma)
    weights = np.ones(mu.shape[0])/mu.shape[0]
    data_gmm = gmm.GMM(mu, sigma, weights)

    mutual_info = 0
    for slabel in range(sensitive_num_classes):
        means, covars = [], []
        counter = 0
        for i in range(s.shape[0]):
            if s[i] == slabel:
                means.append(mu[i])
                covars.append(sigma[i])
                counter += 1
        means, covars = np.array(means), np.array(covars)
        weights = np.ones(means.shape[0])/means.shape[0]
        s_data_gmm = gmm.GMM(means, covars, weights)

        samples = s_data_gmm.sample(n)
        mutual_info += 1./n * np.sum(s_data_gmm.log_likelihood(samples) - data_gmm.log_likelihood(samples)) * counter

    return mutual_info/float(s.shape[0])


def fair_pred_mutual_information(output, s, sensitive_num_classes, T=0.1):

    mutual_info = 0
    gumbel_pred = F.gumbel_softmax(output, tau=T)
    class_prob = torch.sum(gumbel_pred, dim=0)/torch.sum(gumbel_pred)

    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        total_slabel = torch.sum(slabel_mask)
        slabel_pred = slabel_mask.unsqueeze(1)*gumbel_pred
        cond_prob = torch.sum(slabel_pred, dim=0)/(torch.sum(slabel_pred)+ 1e-8)

        mutual_info += torch.sum(cond_prob*torch.log(cond_prob/(class_prob + 1e-8) + 1e-8))

    return mutual_info/float(sensitive_num_classes)


def fairness_metric(sensitive_class_count, sensitive_num_classes):

    sensitive_total_count = [0]*len(sensitive_class_count[0])
    sensitive_class_freq = {}
    for slabel in range(sensitive_num_classes):
        sensitive_class_freq[slabel] = np.array(sensitive_class_count[slabel])/sum(sensitive_class_count[slabel])
        sensitive_total_count = [sensitive_class_count[slabel][i]+sensitive_total_count[i] for i in range(len(sensitive_total_count))]

    sensitive_total_freq = np.array(sensitive_total_count)/sum(sensitive_total_count)

    mutual_info = 0
    for slabel in range(sensitive_num_classes):
        mutual_info += np.sum(sensitive_class_freq[slabel] * np.log(sensitive_class_freq[slabel]/(sensitive_total_freq+1e-8)+1e-8))

    return sensitive_class_freq, mutual_info/float(sensitive_num_classes)

def spd_metric(sensitive_class_freq, sensitive_num_classes):
    total_spd = 0
    max_spd = 0
    count = 0

    sensitive_class_freq_list = []
    for slabel in range(sensitive_num_classes):
        sensitive_class_freq_list.append(sensitive_class_freq[slabel])
    # sensitive_class_freq_array = np.concatenate(sensitive_class_freq_list, axis=0)

    for i in range(sensitive_num_classes):
        for j in range(i+1, sensitive_num_classes):
            abs_spd_arr = np.abs(sensitive_class_freq_list[i] - sensitive_class_freq_list[j])
            if np.max(abs_spd_arr) > max_spd:
                max_spd = np.max(abs_spd_arr)
            total_spd += np.sum(abs_spd_arr)
            count += abs_spd_arr.shape[0]

    return max_spd, total_spd/count




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

    print(data.shape)

    # print(vocabulary.itos)

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    for idx in range(data.shape[1]):
        textidxlist = data[:, idx].tolist()
        textlist.append([vocabulary.itos[int(textidxlist[i])] for i in range(len(textidxlist))])

    return textlist


def store_reconstructed_data(args, X, X_rec):
    if args.dataset == 'utkface' or args.dataset == 'mnist':
        np.save(os.path.join(args.out_dir, args.attack_model_name +
                    '_original_data.npy'), X.clone().detach().cpu().numpy())
        np.save(os.path.join(args.out_dir, args.attack_model_name +
                    '_reconstruct_data.npy'), X_rec.clone().detach().cpu().numpy())
    elif args.dataset == 'twitter':
        Xlist = numpy_to_textlist(X.clone().detach().cpu().numpy())
        Xreclist = numpy_to_textlist(X_rec.clone().detach().cpu().numpy())
        with open(os.path.join(args.out_dir,
            args.attack_model_name + '_original_data.pkl'), 'wb') as fid:
            pickle.dump(Xlist, fid)
        with open(os.path.join(args.out_dir,
            args.attack_model_name + '_reconstruct_data.pkl'), 'wb') as fid:
            pickle.dump(Xreclist, fid)

def info_loss(MI, x, z, s, x_prime, sensitive_num_classes):
    s_onehot = one_hot_encoding(s, sensitive_num_classes).to(z.device)
    Ej = -F.softplus(-MI(x, z, s_onehot)).mean()
    Em = F.softplus(MI(x_prime, z, s_onehot)).mean()
    return Ej - Em

def model_memory(models):
    mem = 0
    for model in models:
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem += mem_params + mem_bufs
    return mem
