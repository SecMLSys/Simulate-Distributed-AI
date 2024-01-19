import torch
from torch import nn
from torch.nn import functional as F

class mlp_decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(mlp_decoder, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        hidden_dims = [in_dim] + hidden_dims

        # Build Decoder
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

        self.pre_decoder = nn.Sequential(*modules)
        ## since all the data is normalized into [0, 1]
        self.decoding = nn.Sequential(nn.Linear(hidden_dims[-1], out_dim), nn.Sigmoid())

    def forward(self, input):
        return self.decoding(self.pre_decoder(input))



class vgg_decoder(nn.Module):
    def __init__(self, img_dim=50, img_channels=3):
        super(vgg_decoder, self).__init__()

        ## the decoder's hyperparameters correspond to the encoder's hyperparameters
        kernel_sizes = [3, 3, 3]
        stride = 1
        upsample_sizes = [(img_dim-2)//2, img_dim-2]

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_sizes[0], stride=stride)
        self.upsample1 = nn.Upsample(size=upsample_sizes[0], mode='bicubic')
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_sizes[1], stride=stride)
        self.upsample2 = nn.Upsample(size=upsample_sizes[1], mode='bicubic')
        self.decoding = nn.Sequential(nn.ConvTranspose2d(32, img_channels, kernel_size=kernel_sizes[2], stride=stride),
                                      nn.Sigmoid())


    def forward(self, x):

        out = F.relu(self.deconv1(x))
        out = self.upsample1(out)
        out = F.relu(self.deconv2(out))
        out = self.upsample2(out)

        return self.decoding(out)



class lstm_decoder(nn.Module):
    def __init__(self, voc_size, hidden_dim=256):

        super(lstm_decoder, self).__init__()

        self.voc_size = voc_size

        self.lstm1 = nn.LSTM(hidden_dim, 2*hidden_dim, batch_first=True)

        self.lstm2 = nn.LSTM(2*hidden_dim, 2*hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)

        self.fc2 = nn.Linear(2*hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # self.lstm1.flatten_parameters()
        # self.lstm2.flatten_parameters()
        lstm1_out, lstm1_hidden = self.lstm1(x)
        lstm2_out, lstm2_hidden = self.lstm2(lstm1_out)

        output = []
        for idx in range(lstm2_out.size(1)):
            out_idx = self.fc2(F.relu(self.fc1(lstm2_out[:, idx, :])))
            output.append(self.sigmoid(out_idx)*(self.voc_size-1))

        output = torch.cat(output, dim=1).squeeze().transpose_(1, 0)

        return output


class resnet_decoder(nn.Module):
    def __init__(self, img_dim=50, img_channels=3):
        super(resnet_decoder, self).__init__()

        ## the decoder's hyperparameters correspond to the encoder's hyperparameters
        kernel_sizes = [3, 3]
        stride = 1

        self.deconv1 = nn.ConvTranspose2d(64, 128, kernel_size=kernel_sizes[0], stride=stride)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=kernel_sizes[0], stride=stride)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_sizes[1], stride=stride)
        self.decoding = nn.Sequential(nn.Conv2d(32, img_channels, kernel_size=kernel_sizes[1], stride=stride),
                                      nn.Sigmoid())


    def forward(self, x):

        out = F.relu(self.deconv1(x))
        out = self.conv1(out)
        out = F.relu(self.deconv2(out))

        return self.decoding(out)


class health_decoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=71):
        super(health_decoder, self).__init__()
        self.decoding = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.BatchNorm1d(in_dim*2),
            nn.ReLU(),
            nn.Linear(in_dim*2, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
            )

    def forward(self, z):
        x = self.decoding(z)
        return x
        