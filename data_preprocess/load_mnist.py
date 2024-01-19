import torch
from torchvision import datasets
from torchvision import transforms

def load_mnist(args, attack=False):

    mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../mnist-data", train=False, download=True, transform=transforms.ToTensor())

    if attack:
        train_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader
