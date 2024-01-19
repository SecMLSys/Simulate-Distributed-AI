import torch
from torch import nn
from torch.nn import functional as F


class mlp_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(mlp_encoder, self).__init__()
        
        n_h = 128
        modules = []
        if hidden_dims is None:
            hidden_dims = [n_h]

        hidden_dims = [in_dim] + hidden_dims

        # Build Encoder
        for layer_idx in range(len(hidden_dims)):
            if bn:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.BatchNorm1d(hidden_dims[layer_idx+1]),
                        nn.ReLU(),
                        nn.Dropout(drop_rate))
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.ReLU(),
                        nn.Dropout(drop_rate))
                )

        self.encoder = nn.Sequential(*modules)

    def forward(self, input):
        encoding = self.encoder(input)
        return encoding

        

class vgg_encoder(nn.Module):
    def __init__(self, img_channels=3):
        super(vgg_encoder, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        

    def forward(self, x):
        encoding = self.convnet(x)
        return encoding


class lstm_encoder(nn.Module):
    def __init__(self, voc_size, embedding_dim=32, hidden_dim=256):

        super(lstm_encoder, self).__init__()

        self.embedding = nn.Embedding(voc_size, embedding_dim)

        self.encoding = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        
    def forward(self, x):
        x = torch.t(x)

        embedded = self.embedding(x)

        #embedded = [batch size, sent len, emb dim]

        en, (hn, cn) = self.encoding(embedded)

        #output = [batch size, sent len, hid dim]
        #hn = [batch size, 1, hid dim]
        try:
            assert torch.equal(en[:, -1,:], hn.squeeze(0))
        except:
            # print(en.size(), hn.size())
            # print(en[:, -1,:], hn.squeeze(0))
            exit()

        return en
        



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out



class resnet_encoder(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], img_channels=3, num_classes=10):
        super(resnet_encoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.encoding = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        en = F.relu(self.encoding(out))

        return en




class purchase_encoder(nn.Module):
    def __init__(self):
        super(purchase_encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600,1024),
            nn.Tanh())

        self.encoding = nn.Sequential(
            nn.Linear(1024,512),
            nn.Tanh())


    def forward(self, x, mask=None):
        x = self.features(x)
        en = self.encoding(x)
        return en
        


class texas_encoder(nn.Module):
    def __init__(self):
        super(texas_encoder, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh())
        self.encoding = nn.Sequential(
            nn.Linear(1024,512),
            nn.Tanh())


    def forward(self, x, mask=None):
        x = self.features(x)
        en = self.encoding(x)
        return en
       
            

class health_encoder(nn.Module):
    def __init__(self, input_dim=71):
        super(health_encoder, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.encoding = nn.Sequential(
            nn.Linear(128, 256))


    def forward(self, x):
        x = self.features(x)
        en = self.encoding(x)
        return en


