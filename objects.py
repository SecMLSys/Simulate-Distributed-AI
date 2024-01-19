import os
import socket
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
import logging

import sys
sys.path.insert(1, '../')
from data_preprocess.config import dataset_class_params




class Server:
    def __init__(self, args):
        target_num_classes = dataset_class_params[args.dataset][args.target_attr]

        self.classifier = load_classifier(args)

        if args.use_cuda:
            self.classifier = self.classifier.cuda()

        if args.test:
            self.classifier.eval()
            state_dict = torch.load(os.path.join(args.out_dir,
                            args.model_name + '_classifier_model.pth'))
            self.classifier.load_state_dict(state_dict)
        else:
            self.classifier.train()
            self.opt_classifier = torch.optim.Adam(self.classifier.parameters(), 
                                            lr=args.server_lr, weight_decay=args.weight_decay)

        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda

    def model_update(self, client_output, label):

        if self.use_cuda:
            client_output = client_output.cuda()
            label = label.cuda()

        output = self.classifier(client_output)
        loss = self.criterion(output, label)

        self.opt_classifier.zero_grad()
        loss.backward()
        self.client_output_grad = client_output.grad.clone().detach()
        self.opt_classifier.step()

        return output, loss

    def prediction(self, client_output):
        if self.use_cuda:
            client_output = client_output.cuda()
        output = self.classifier(client_output)
        return output.clone().detach()

    def create_socket(self, args):
        print('finish setting up the local host', flush=True)
        self.sock = socket.socket()
        self.sock.bind((args.hostname, args.port))
        self.sock.listen(5)

    def accept_clients(self, args):
        self.client_sock_list = []
        for user in range(args.num_users):
            conn, addr = self.sock.accept()
            print('Conntected with: {}'.format(addr))
            self.client_sock_list.append(conn)

    def send_msg(self, client_id, msg):
        send_msg(self.client_sock_list[client_id], msg)

    def recv_msg(self, client_id):
        return recv_msg(self.client_sock_list[client_id])

    def save_model(self, args):
        torch.save(self.classifier.state_dict(), os.path.join(args.out_dir,
                                                    args.model_name + '_classifier_model.pth'))


class Client:
    def __init__(self, args):

        self.train_loader, self.test_loader = load_data(args)
        self.feature_learner = load_encoder(args)

        ### use cuda
        if args.use_cuda:
            self.feature_learner = self.feature_learner.cuda()

        if args.test or args.privacy:
            self.feature_learner.eval()
            state_dict = torch.load(os.path.join(args.out_dir, args.model_name + '_feature_model.pth'))
            self.feature_learner.load_state_dict(state_dict)
        
        else:
            self.feature_learner.train()
            self.opt_feature = torch.optim.Adam(self.feature_learner.parameters(), lr=args.lr, 
                                                    weight_decay=args.weight_decay)
            print('model size: ', model_memory([self.feature_learner]))

        if args.privacy:
            self.privacy_criterion = nn.CrossEntropyLoss()
            self.privacy_examiner = load_privacy_examiner(args)
            self.opt_privacy = torch.optim.Adam(self.privacy_examiner.parameters(), lr=args.lr, 
                                                    weight_decay=args.weight_decay)

        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.use_cuda = args.use_cuda

    def model_forward(self, X):

        self.z = self.feature_learner(X)
        return self.z

    def send_msg(self, msg):
        send_msg(self.server_socket, msg)

    def recv_msg(self):
        return recv_msg(self.server_socket)


    def model_backward(self, z_grad):

        if self.use_cuda:
            z_grad = z_grad.cuda()

        loss = torch.sum(self.z * z_grad.detach())

        # print(loss)
        self.opt_feature.zero_grad()
        loss.backward()
        self.opt_feature.step()

        # check_model_params(self.feature_learner)

    def connect_server(self, args):
        self.server_socket = socket.socket()
        self.server_socket.connect((args.hostname, args.port))
        return self.server_socket

    def save_model(self, args):
        torch.save(self.feature_learner.state_dict(), os.path.join(args.out_dir,
                                                    args.model_name + '_feature_model.pth'))
        
    def train_privacy_examiner(self, s):

        output = self.privacy_examiner(self.z.detach())
        loss = self.privacy_criterion(output, s)

        self.opt_privacy.zero_grad()
        loss.backward()
        self.opt_privacy.step()

        return output, loss
    
    def save_privacy_model(self, args):
        torch.save(self.privacy_examiner.state_dict(), os.path.join(args.out_dir,
                                                    args.model_name + '_privacy_examiner.pth'))


class TrainingMetrics:
    def __init__(self, args):
        self.criterion = nn.CrossEntropyLoss()
        self.create_logger(args)

    def initialize_metrics(self):
        self.start_epoch_time = time.time()
        self.train_normal_loss = 0
        self.train_normal_acc = 0
        self.train_n = 0

    def create_logger(self, args):
        self.logger = logging.getLogger('training_metrics')
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        logfile = os.path.join(args.out_dir, args.model_name + '_train_output.log')

        if os.path.exists(logfile):
            os.remove(logfile)

        logging.basicConfig(format='[%(asctime)s] - %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.INFO,
                            filename=logfile)
        self.logger.info(args)

    def update(self, pred, label, loss):
        self.train_n += label.size(0)
        self.train_normal_acc += (pred.max(1)[1] == label).sum().item()
        self.train_normal_loss += loss.item()*label.size(0)

    def log_metrics(self, epoch):
        epoch_time = time.time() - self.start_epoch_time
        avg_acc = self.train_normal_acc/self.train_n
        avg_loss = self.train_normal_loss/self.train_n
        print('Epoch: %d \t Train Acc: %.4f \t Train Loss: %.4f'%(epoch, avg_acc, avg_loss))
        self.logger.info('%d \t\t %.4f \t\t %.4f', epoch, avg_acc, avg_loss)



class TrainingEvaluationMetrics:
    def __init__(self, args, fairness=False):
        self.train_normal_loss = 0
        self.train_normal_acc = 0
        self.train_n = 0
        self.target_num_classes = dataset_class_params[args.dataset][args.target_attr]

        if fairness:
            self.sensitive_num_classes = dataset_class_params[args.dataset][args.sensitive_attr]
            self.sensitive_class_count = {}
            for slabel in range(sensitive_num_classes):
                self.sensitive_class_count[slabel] = [0 for tlabel in range(self.target_num_classes)]

        self.criterion = nn.CrossEntropyLoss()
        self.fairness = fairness

        self.create_logger(args)


    def create_logger(self, args):
        self.logger = logging.getLogger('training_evaluation_metrics')
        logfile = os.path.join(args.out_dir, args.model_name + '_train_evaluation_output.log')
        if os.path.exists(logfile):
            os.remove(logfile)

        logging.basicConfig(format='[%(asctime)s] - %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.INFO,
                            filename=logfile)
        self.logger.info(args)

    def update(self, pred, y, s=None):
        celoss = self.criterion(pred, y)
        self.train_n += y.size(0)
        self.train_normal_acc += (pred.max(1)[1] == y).sum().item()
        self.train_normal_loss += celoss.item()*y.size(0)

        if self.fairness:
            pred_labels = pred.max(1)[1]
            for idx in range(s.size(0)):
                self.sensitive_class_count[s[idx].item()][pred_labels[idx].item()] += 1


    def summary(self):
        avg_acc = self.train_normal_acc/self.train_n
        avg_loss = self.train_normal_loss/self.train_n
        print('Normal Train Acc: %.4f \t Normal Train Loss: %.4f'%(avg_acc, avg_loss))
        self.logger.info('Normal Train Acc: %.4f \t Normal Train Loss: %.4f', avg_acc, avg_loss)

        if self.fairness:
            sensitive_class_freq, fairMI = fairness_metric(self.sensitive_class_count,
                                                            self.sensitive_num_classes)
            max_spd, avg_spd = spd_metric(sensitive_class_freq,
                                            self.sensitive_num_classes)

            for slabel in range(self.sensitive_num_classes):
                class_log_info = '['
                for i in range(self.target_num_classes-1):
                    class_log_info += str(sensitive_class_freq[slabel][i]) + ', '
                class_log_info += str(sensitive_class_freq[slabel][self.target_num_classes-1]) + ']'
                logger.info('sensitive class: %d:\t %s', slabel, class_log_info)

            self.logger.info('fairness metric: %.6f', fairMI)
            self.logger.info('Max SPD: %.6f, Avg SPD: %.6f', max_spd, avg_spd)
            print('fairness metric: {0:.6f}, Max SPD: {0:.6f}, Avg SPD: {0:.6f}'.format(fairMI, max_spd, avg_spd))



class EvaluationMetrics:
    def __init__(self, args, fairness=False):
        self.test_normal_loss = 0
        self.test_normal_acc = 0
        self.test_n = 0
        self.target_num_classes = dataset_class_params[args.dataset][args.target_attr]

        if fairness:
            self.sensitive_num_classes = dataset_class_params[args.dataset][args.sensitive_attr]
            self.sensitive_class_count = {}
            for slabel in range(sensitive_num_classes):
                self.sensitive_class_count[slabel] = [0 for tlabel in range(self.target_num_classes)]

        self.criterion = nn.CrossEntropyLoss()
        self.fairness = fairness

        self.create_logger(args)


    def create_logger(self, args):
        self.logger = logging.getLogger('evaluation_metrics')
        logfile = os.path.join(args.out_dir, args.model_name + '_test_output.log')
        if os.path.exists(logfile):
            os.remove(logfile)

        logging.basicConfig(format='[%(asctime)s] - %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.INFO,
                            filename=logfile)
        self.logger.info(args)

    def update(self, pred, y, s=None):
        celoss = self.criterion(pred, y)
        self.test_n += y.size(0)
        self.test_normal_acc += (pred.max(1)[1] == y).sum().item()
        self.test_normal_loss += celoss.item()*y.size(0)

        if self.fairness:
            pred_labels = pred.max(1)[1]
            for idx in range(s.size(0)):
                self.sensitive_class_count[s[idx].item()][pred_labels[idx].item()] += 1


    def summary(self):
        avg_acc = self.test_normal_acc/self.test_n
        avg_loss = self.test_normal_loss/self.test_n
        print('Normal Test Acc: %.4f \t Normal Test Loss: %.4f'%(avg_acc, avg_loss))
        self.logger.info('Normal Test Acc: %.4f \t Normal Test Loss: %.4f', avg_acc, avg_loss)

        if self.fairness:
            sensitive_class_freq, fairMI = fairness_metric(self.sensitive_class_count,
                                                            self.sensitive_num_classes)
            max_spd, avg_spd = spd_metric(sensitive_class_freq,
                                            self.sensitive_num_classes)

            for slabel in range(self.sensitive_num_classes):
                class_log_info = '['
                for i in range(self.target_num_classes-1):
                    class_log_info += str(sensitive_class_freq[slabel][i]) + ', '
                class_log_info += str(sensitive_class_freq[slabel][self.target_num_classes-1]) + ']'
                logger.info('sensitive class: %d:\t %s', slabel, class_log_info)

            self.logger.info('fairness metric: %.6f', fairMI)
            self.logger.info('Max SPD: %.6f, Avg SPD: %.6f', max_spd, avg_spd)
            print('fairness metric: {0:.6f}, Max SPD: {0:.6f}, Avg SPD: {0:.6f}'.format(fairMI, max_spd, avg_spd))


def check_model_params(model):
    for p in model.parameters():
        if p.requires_grad:
             print(p.name, p.data)
        break
