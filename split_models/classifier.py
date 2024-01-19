import torch
from torch import nn
from torch.nn import functional as F


class mlp_classifier(nn.Module):
    def __init__(self, in_dim, hidden_dims=None, bn=True, drop_rate=0.0, num_classes=2):
        super(mlp_classifier, self).__init__()

        self.drop_rate = drop_rate
        modules = []
        if hidden_dims is None:
            hidden_dims = [in_dim*2]

        hidden_dims = [in_dim] + hidden_dims

        for layer_idx in range(len(hidden_dims)-1):
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



        self.features = None if len(modules) == 0 else nn.Sequential(*modules)
        self.logits = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, input):
        features = F.dropout(input, p=self.drop_rate, training=self.training)
        if self.features is not None: features = self.features(features)
        return self.logits(features)



class binary_classifier(nn.Module):
    def __init__(self, in_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(binary_classifier, self).__init__()

        self.drop_rate = drop_rate
        modules = []
        if hidden_dims is None:
            hidden_dims = []

        hidden_dims = [in_dim] + hidden_dims

        for layer_idx in range(len(hidden_dims)-1):
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



        self.features = None if len(modules) == 0 else nn.Sequential(*modules)
        self.logit = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        features = F.dropout(input, p=self.drop_rate, training=self.training)
        if self.features is not None: features = self.features(features)
        return self.sigmoid(self.logit(features))



class vgg_classifier(nn.Module):
    def __init__(self, img_dim=50, num_classes=2):
        super(vgg_classifier, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        fea_dim = ((((img_dim//2)//2)//2)//2)//2

        self.fcnet = nn.Sequential(
            nn.Linear(512 * fea_dim * fea_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(out.size(0), -1)
        out = self.fcnet(out)
        return out



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
        out = F.relu(out)
        return out


class resnet_classifier(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=2):
        super(resnet_classifier, self).__init__()
        self.in_planes = 64

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class lstm_classifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=10):
        super(lstm_classifier, self).__init__()
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm1_out, lstm1_hidden = self.lstm1(x)
        lstm2_out, lstm2_hidden = self.lstm2(lstm1_out)
        out = torch.stack([lstm2_out[i, -1] for i in range(len(lstm2_out))], dim=0)
        out = self.fc(out)
        return out


class purchase_classifier(nn.Module):
    def __init__(self, num_classes=100):
        super(purchase_classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)


    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


class texas_classifier(nn.Module):
    def __init__(self, num_classes=100):
        super(texas_classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)


    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


class health_classifier(nn.Module):
    def __init__(self, num_classes):
        super(health_classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128,num_classes)


    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)
