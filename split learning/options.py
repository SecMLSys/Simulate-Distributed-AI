import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_users', default=1, type=int)
    parser.add_argument('--hostname', default='localhost', type=str)
    parser.add_argument('--port', default=1024, type=int)

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-epochs', default=50, type=int)

    parser.add_argument('--dataset', default='health', type=str, choices=['mnist', 'purchase', 'texas', 'health', 'utkface', 'twitter'])
    parser.add_argument('--sensitive-attr', default='gender', type=str, choices=['age', 'gender', 'race', 'identity'])
    parser.add_argument('--target-attr', default='age', type=str, choices=['age', 'charlson', 'style', 'type', 'digit'])
    
    parser.add_argument('--surrogate', default='ce', type=str)
    parser.add_argument('--model-architecture', default='vgg', type=str, choices=['vgg', 'resnet', 'mlp', 'lstm'])

    ## follow the existing work to train gan, we use adam optimizer
    parser.add_argument("--server-lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help='weight decay')

    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--privacy', action='store_true')

    parser.add_argument('--out-dir', default='split_learning', type=str)

    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()