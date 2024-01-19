import time
from tqdm import tqdm

import torch

from options import get_args
from utils import set_random_seed
from objects import Server, Client
from objects import TrainingMetrics, TrainingEvaluationMetrics, EvaluationMetrics



def train(args):

    client = Client(args)
    server = Server(args)

    metrics = TrainingMetrics(args)

    ### start training
    start_train_time = time.time()
    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        metrics.initialize_metrics()
        with tqdm(total=len(client.train_loader)) as pbar:
            for i, (X, y, s) in enumerate(client.train_loader):
                if args.use_cuda:
                    X, y, s = X.cuda(), y.cuda(), s.cuda()

                z = client.model_forward(X)
                ## send client output
                client_output = z.clone().detach().requires_grad_(True)
                ## receive client output
                pred, loss = server.model_update(client_output, y)

                metrics.update(pred, y, loss)

                ## recent grad from the server
                z_grad = server.client_output_grad.clone().detach()
                client.model_backward(z_grad)

                pbar.update(1)

        metrics.log_metrics(epoch)

    client.save_model(args)
    server.save_model(args)

    train_time = time.time() - start_train_time
    print("total train time {}".format(train_time))


def evaluate(args):

    client = Client(args)
    server = Server(args)

    metrics = TrainingEvaluationMetrics(args)

    for i, data in enumerate(client.train_loader):
        X, y = data[0], data[1]
        if args.use_cuda:
            X, y = X.cuda(), y.cuda()

        z = client.model_forward(X)
        ## send client output
        client_output = z.clone().detach()

        pred = server.prediction(client_output)

        metrics.update(pred, y)

    metrics.summary()


    metrics = EvaluationMetrics(args)

    for i, data in enumerate(client.test_loader):
        X, y = data[0], data[1]
        if args.use_cuda:
            X, y = X.cuda(), y.cuda()

        z = client.model_forward(X)
        ## send client output
        client_output = z.clone().detach()

        pred = server.prediction(client_output)

        metrics.update(pred, y)

    metrics.summary()


def privacy_test(args):

    client = Client(args)
    
    metrics = TrainingMetrics(args)

    ### start training
    start_train_time = time.time()
    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        metrics.initialize_metrics()
        with tqdm(total=len(client.train_loader)) as pbar:
            for i, (X, y, s) in enumerate(client.train_loader):
                if args.use_cuda:
                    X, y, s = X.cuda(), y.cuda(), s.cuda()

                client.model_forward(X)
               
                pred, loss = client.train_privacy_examiner(s)

                metrics.update(pred, y, loss)

                pbar.update(1)

        metrics.log_metrics(epoch)

    client.save_model(args)

    train_time = time.time() - start_train_time
    print("total train time {}".format(train_time))



if __name__ == '__main__':
    args = get_args()

    args.out_dir = args.dataset + '_results'

    args.model_name = 'seed_{}_epochs_{}_target_{}'.format(args.seed, args.num_epochs, args.target_attr)

    args.model_name += '_sensitive_{}'.format(args.sensitive_attr)
        

    ### random seeds
    torch.backends.cudnn.deterministic = True
    set_random_seed(args)

    if args.test:
        evaluate(args)
    elif args.privacy:
        privacy_test(args)
    else:
        train(args)
