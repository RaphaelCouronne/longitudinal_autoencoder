import torch.nn as nn
from src.models.networks import MetaActivation
from functools import reduce
from operator import mul

############
#### 2D
############


class DeconvBN2d_field(nn.Module):
    """
    Velocity Field Decoder with grid size:
    * kernel_size = 2
    * 4 levels
    * Batch-Norm layers
    """

    def __init__(self, in_dim, out_grid_size, out_channels=2, intermediate_activation='lrelu',
                 final_activation='linear', bias=True):
        nn.Module.__init__(self)
        self.inner_grid_size = [int(elt * 2 ** -5) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = in_dim
        self.bias = bias

        # Architecture
        self.linear1 = nn.Sequential(nn.Linear(self.latent_dimension,  32 * self.n, bias=self.bias),
                                     nn.Tanh())
        self.linear2 = nn.Sequential(nn.Linear(32 * self.n, 32 * self.n, bias=self.bias),
                                     nn.Tanh())

        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm2d(32),
            MetaActivation(intermediate_activation),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm2d(16),
            MetaActivation(intermediate_activation),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm2d(8),
            MetaActivation(intermediate_activation),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm2d(4),
            MetaActivation(intermediate_activation),
            nn.ConvTranspose2d(4, out_channels, kernel_size=2, stride=2, padding=0, bias=self.bias),
            MetaActivation(final_activation)
        ])

        print('>> DeconvBN2d_field has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        expanded_size = tuple([bts, 32] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.linear2(x).view(expanded_size)
        for layer in self.layers:
            x = layer(x)
        return x


############
#### 3D
############


class DeconvBN3d_field(nn.Module):
    """
    Velocity Field Decoder with grid size:
    * kernel_size = 2
    * 4 levels
    * Batch-Norm layers
    """

    def __init__(self, in_dim, out_grid_size, out_channels=3, intermediate_activation='tanh',
                 final_activation='linear', bias=True):
        nn.Module.__init__(self)
        self.inner_grid_size = [int(elt * 2 ** -5) for elt in out_grid_size]
        self.n = int(reduce(mul, self.inner_grid_size))
        self.latent_dimension = in_dim
        self.bias = bias

        # Architecture
        self.linear1 = nn.Sequential(nn.Linear(self.latent_dimension,  64 * self.n, bias=self.bias),
                                     MetaActivation(intermediate_activation))
        self.linear2 = nn.Sequential(nn.Linear(64 * self.n, 64 * self.n, bias=self.bias),
                                     MetaActivation(intermediate_activation))

        self.layers = nn.ModuleList([
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(8, 4, kernel_size=2, stride=2, padding=0, bias=self.bias),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4, out_channels, kernel_size=2, stride=2, padding=0, bias=self.bias),
            MetaActivation(final_activation)
        ])

        print('>> DeconvBN3d_field has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        bts = x.size(0)
        expanded_size = tuple([bts, 64] + self.inner_grid_size)
        x = self.linear1(x)
        x = self.linear2(x).view(expanded_size)
        for layer in self.layers:
            x = layer(x)
        return x

