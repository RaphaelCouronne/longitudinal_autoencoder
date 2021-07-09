import torch.nn as nn
import torch

############
#### 1D
############


class Encoding_1D(nn.Module):
    """
    3 Dimensional Dimensionality Reduction via Convolutions
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Encoding_1D, self).__init__()
        """
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, pre_encoder_dim),
        )"""

        """
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, pre_encoder_dim),
        )
        """

        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, out_dim)
        self.non_linearity = nn.LeakyReLU()
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)

        self.out_dim = out_dim
        print('Encoder 1D has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # Convolutions
        #for layer in self.layers:
        #    x = layer(x)

        x = self.fc_1(x)
        x = self.bn_2(self.fc_2(self.non_linearity(x)))+x
        x = self.fc_3(self.non_linearity(x))

        return x


############
#### 2D
############
class Convolutions_2D_64(nn.Module):
    """
    3 Dimensional Dimensionality Reduction via Convolutions
    """
    def __init__(self):
        super(Convolutions_2D_64, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 8, 4, 4),  # 8 x 16 x 16
                                     nn.BatchNorm2d(8),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(8, 16, 2, 2),  # 16 x 8 x 8
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(16, 16, 2, 2),  # 16 x 4 x 4
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU(),
                                     #nn.Conv2d(16, 16, 2, 2),  # 16 x 2 x 2
                                     nn.Conv2d(16, 32, 2, 2),  # 16 x 2 x 2
                                     #nn.LeakyReLU(),
                             )

        self.out_dim = 64*2
        print('Encoder 2D 64x64 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # Convolutions
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def out_dim():
        return 64


class Convolutions_2D_128(nn.Module):
    """
    3 Dimensional Dimensionality Reduction via Convolutions
    """
    def __init__(self):
        super(Convolutions_2D_128, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),
            #nn.LeakyReLU()
        )

        self.out_dim = 256*2
        print('Encoder 2D 64x64 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # Convolutions
        for layer in self.layers:
            x = layer(x)
        return x


############
#### 3D
############


class Convolutions_3D_64(nn.Module):
    """
    3 Dimensional Dimensionality Reduction via Convolutions
    """
    def __init__(self):
        super(Convolutions_3D_64, self).__init__()
        self.layers = nn.Sequential(nn.Conv3d(1, 8, 4, 4),  # 8 x 16 x 16 x 16
                                    nn.BatchNorm3d(8),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(8, 16, 2, 2),  # 16 x 8 x 8 x 8
                                    nn.BatchNorm3d(16),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(16, 16, 2, 2),  # 16 x 4 x 4 x 4
                                    nn.BatchNorm3d(16),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(16, 16, 2, 2),  # 16 x 2 x 2 x 2
                                     nn.LeakyReLU(),
                                     )

        self.out_dim = 128
        print('Encoder 3D 64x64x64 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))
    def forward(self, x):
        # Convolutions
        for layer in self.layers:
            x = layer(x)
        return x


############
#### High capacity Networks
############

class Convolutions_3D_64_Big(nn.Module):
    def __init__(self, y_dim=0,input_channel_size=1,device ='cpu'):
        super(Convolutions_3D_64_Big, self).__init__()
        #self.z_dim = z_dim
        self.y_dim = y_dim
        self.device = device
        self.net = nn.Sequential(
            nn.Conv3d(input_channel_size,16,3,padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        #self.Linear1 = nn.Linear(1024,512).to(self.device)
        #self.Linear2 = nn.Linear(1024,1024).to(self.device)
        #self.Linear3 = nn.Linear(1024,2*z_dim).to(self.device)
        self.net = self.net.float()
        #self.dropout1 = nn.Dropout(p=0.5)
        #self.dropout2 = nn.Dropout(p=0.5)
        #self.Linear4 = nn.Linear(1024,z_dim).to(self.device)
        self.out_dim = 1024

    def forward(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy.float())
        return h


#%%


class Convolutions_3D_DAT(nn.Module):
    """
    3 Dimensional Dimensionality Reduction via Convolutions
    """
    def __init__(self):
        super(Convolutions_3D_DAT, self).__init__()
        self.layers = nn.Sequential(nn.Conv3d(1, 8, 4, 4),  # 8 x 16 x 16 x 16
                                    nn.BatchNorm3d(8),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(8, 16, 2, 2),  # 16 x 8 x 8 x 8
                                    nn.BatchNorm3d(16),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(16, 32, 2, 2),  # 16 x 4 x 4 x 4
                                    nn.BatchNorm3d(32),
                                     nn.LeakyReLU(),
                                     nn.Conv3d(32, 64, 2, 2),  # 16 x 2 x 2 x 2
                                     nn.LeakyReLU(),
                                     )

        self.out_dim = 64*2*3*2
        print('Encoder 3D 64x64x64 has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))
    def forward(self, x):
        # Convolutions
        for layer in self.layers:
            x = layer(x)
        return x


#%%

############
#### Linear Modules
############

"""
NB : in practice
in_dim = 64 for 2D convolutions, 128 for 3D convolutions
hidden_dim = 32
out_dim = pre_encoder_dim = 12 (more might be required for 2D/3D)

1 MLP_2 for space
1 MLP_2 for time

then for each a linear layer to get mu, logvar
"""

class MLP_2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP_2, self).__init__()
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        self.non_linearity = nn.Tanh()

    def forward(self, x):
        x = self.non_linearity(x)
        x = self.fc_1(x)
        x = self.non_linearity(x)
        x = self.fc_2(x)
        return x

class MLP_variational(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_variational, self).__init__()
        out_dim = out_dim
        self.fc_mu = nn.Linear(in_dim, out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)
        self.non_linearity = nn.LeakyReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.non_linearity(x)
        return self.fc_mu(x), self.fc_logvar(x)


############
#### Reccurent Modules (used in the affine case)
############

class ScalarRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1):
        super(ScalarRNN, self).__init__()
        self.rnn = nn.RNN(in_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        #self.rnn = nn.GRU(input_size=in_dim+1, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)
        #self.rnn = nn.LSTM(input_size=in_dim+1, hidden_size=hidden_dim*2, batch_first=True)
        self.fc_1 = nn.Linear(hidden_dim*num_layers, out_dim)
        self.num_layers = num_layers

    def forward(self, x):
        hidden, last_hidden = self.rnn(x.unsqueeze(0))
        out = self.fc_1(torch.tanh(last_hidden.view(1,-1)))
        return out
