import torch.nn as nn
from src.models.networks import MetaActivation
import torch

############
#### 1D
############


class Decoder1D(nn.Module):
    """
    1 Dimensional Decoder architecture
    """
    def __init__(self, in_dim, out_dim, hidden_dim=16, bias=True, last_function='identity'):
        super(Decoder1D, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=bias),
                                     nn.BatchNorm1d(num_features=hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim, bias=bias),
                                     nn.BatchNorm1d(num_features=hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, out_dim, bias=bias),
                                     #nn.Tanh(),
                                     #nn.Linear(hidden_dim, out_dim, bias=bias),
                                     MetaActivation(last_function),
                                     ])
        print('Scalar decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

############
#### 2D
############

"""
class Deconv2D_64(nn.Module):

    def __init__(self, in_dim, last_function=nn.Identity()):
        self.in_dim = in_dim
        super(Deconv2D_64, self).__init__()
        self.fc_1 = nn.Linear(in_dim, 2*in_dim)
        self.fc_2 = nn.Linear(2*in_dim, 128)
        self.non_linearity = nn.LeakyReLU()
        self.in_dim = in_dim
        self.last_function = last_function#nn.Sigmoid()
        ngf = 2

        self.layers = nn.ModuleList([
            # 1
            nn.ConvTranspose2d(32, 16 * ngf, 3, stride=3),
            nn.BatchNorm2d(16 * ngf),
            nn.LeakyReLU(),
            # 2
            nn.ConvTranspose2d(16 * ngf, 8 * ngf, 3, stride=3, padding=1),
            nn.BatchNorm2d(8 * ngf),
            nn.LeakyReLU(),
            # 3
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 3, stride=2, padding=1),
            nn.BatchNorm2d(4 * ngf),
            nn.LeakyReLU(),
            # 3
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 3, stride=2, padding=0),
            nn.BatchNorm2d(2 * ngf),
            nn.LeakyReLU(),
            # 4
            nn.ConvTranspose2d(2 * ngf, 1, 2, stride=1),
            self.last_function,
            #nn.Conv2d(ngf,1,(2,2),1)
            #nn.Sigmoid()
        ])




        print('Deconv decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # 2 MLP Layers
        x = self.non_linearity(self.fc_1(x))
        x = self.fc_2(x)
        # Reshape
        #x = x.view(-1, 8, 4, 4)
        x = x.view(-1, 32, 2, 2)
        # Deconvolutions
        for layer in self.layers:
            x = layer(x)
        x = x[:,:,:64,:64]
        #x = self.last_function(x)
        return x"""


class Deconv2D_64(nn.Module):

    def __init__(self, in_dim=2, out_channels=1, last_function='identity'):
        super(Deconv2D_64, self).__init__()
        self.in_dim = in_dim
        self.fc = nn.Linear(in_dim, 2*in_dim)
        self.ta = nn.Tanh()
        ngf = 2
        self.in_dim = 2*in_dim
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(self.in_dim, 16 * ngf, 4, stride=4),
            nn.BatchNorm2d(16 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16 * ngf, 8 * ngf, 2, stride=2),
            nn.BatchNorm2d(8 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 2, stride=2),
            nn.BatchNorm2d(4 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
            nn.BatchNorm2d(2 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2 * ngf, out_channels, 2, stride=2),
            MetaActivation(last_function),
        ])
        print('Deconv decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.ta(self.fc(x))
        x = x.view(-1, self.in_dim, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x


class Deconv2D_128(nn.Module):
    """
    2 Dimensional Decoder architecture
    """
    def __init__(self, in_dim, out_channels=1, last_function='identity'):
        self.in_dim = in_dim
        super(Deconv2D_128, self).__init__()
        self.fc_1 = nn.Linear(in_dim, 2*in_dim)
        self.fc_2 = nn.Linear(2*in_dim, 128)
        self.non_linearity = nn.LeakyReLU()
        self.in_dim = in_dim
        ngf = 2

        self.layers = nn.ModuleList([

                         nn.ConvTranspose2d(32, 32 * ngf, 2, stride=2),
            nn.BatchNorm2d(32*ngf),
                         nn.LeakyReLU(),
                         nn.ConvTranspose2d(32*ngf , 16 * ngf, 2, stride=2),
            nn.BatchNorm2d(16 * ngf),
                         nn.LeakyReLU(),
                         nn.ConvTranspose2d(16 * ngf, 8 * ngf, 2, stride=2),
            nn.BatchNorm2d(8 * ngf),
                         nn.LeakyReLU(),
                         nn.ConvTranspose2d(8 * ngf, 4 * ngf, 2, stride=2),
            nn.BatchNorm2d(4 * ngf),
                         nn.LeakyReLU(),
                         nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
            nn.BatchNorm2d(2 * ngf),
                         nn.LeakyReLU(),
                         nn.ConvTranspose2d(2 * ngf, out_channels, 2, stride=2),
            MetaActivation(last_function)
        ])


        """
        
                    # 1
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid(),
        
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(8, 8, 3, stride=3),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=3, padding=2),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, 3, stride=3, padding=0),
            nn.Sigmoid()
        ])"""

        print('Deconv decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # 2 MLP Layers
        x = self.non_linearity(self.fc_1(x))
        x = self.fc_2(x)
        # Reshape
        #x = x.view(-1, 8, 4, 4)
        x = x.view(-1, 32, 2, 2)
        # Deconvolutions
        for layer in self.layers:
            x = layer(x)
        #x = x[:,:,:64,:64]
        #x = self.last_function(x)
        return x

"""
class Deconv2D_128(nn.Module):

     def __init__(self, in_dim=2, last_function=nn.ReLU()):
         self.in_dim = in_dim
         super(Deconv2D_128, self).__init__()
         self.fc = nn.Linear(in_dim, in_dim)
         self.ta = nn.Tanh()
         ngf = 2
         self.in_dim = in_dim
         last_function = nn.ReLU()
         #last_function = nn.LeakyReLU() if last_layer == 'relu' else nn.Tanh()

         self.layers = nn.ModuleList([
             nn.ConvTranspose2d(in_dim, 32 * ngf, 4, stride=4),
             nn.LeakyReLU(),
             nn.ConvTranspose2d(32 * ngf, 16 * ngf, 2, stride=2),
             nn.LeakyReLU(),
             nn.ConvTranspose2d(16 * ngf, 8 * ngf, 2, stride=2),
             nn.LeakyReLU(),
             nn.ConvTranspose2d(8 * ngf, 4 * ngf, 2, stride=2),
             nn.LeakyReLU(),
             nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
             nn.LeakyReLU(),
             nn.ConvTranspose2d(2 * ngf, 1, 2, stride=2),
             last_function
         ])
         print('Deconv decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

     def forward(self, x):
         x = self.ta(self.fc(x))
         x = x.view(-1, self.in_dim, 1, 1)
         for layer in self.layers:
             x = layer(x)
         return x"""


############
#### 3D
############


class Deconv3D_64(nn.Module):
    """
    3 Dimensional Decoder architecture
    """
    def __init__(self, in_dim, out_channels=1, last_function='sigmoid'):
        self.in_dim = in_dim
        super(Deconv3D_64, self).__init__()
        self.fc_1 = nn.Linear(in_dim, in_dim)
        self.fc_2 = nn.Linear(in_dim, 2*in_dim)
        self.non_linearity = nn.Tanh()
        self.in_dim = in_dim
        ngf = 2
        self.layers = nn.ModuleList([
            nn.ConvTranspose3d(2*in_dim, 32 * ngf, 2, stride=2),
            nn.BatchNorm3d(32 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32*ngf, 16 * ngf, 2, stride=2),
            nn.BatchNorm3d(16 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16 * ngf, 8 * ngf, 2, stride=2),
            nn.BatchNorm3d(8 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(8 * ngf, 4 * ngf, 2, stride=2),
            nn.BatchNorm3d(4 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4 * ngf, 4 * ngf, 2, stride=2),
            nn.BatchNorm3d(4 * ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4 * ngf, out_channels, 2, stride=2),
            MetaActivation(last_function),
        ])
        print('Deconv decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # 2 MLP Layers
        x = self.non_linearity(self.fc_1(x))
        x = self.fc_2(x)
        # Reshape
        x = x.view(-1, 2*self.in_dim, 1, 1, 1)
        # Deconvolutions
        for layer in self.layers:
            x = layer(x)
        return x


#%%


class Deconv3D_64_big(nn.Module):
    def __init__(self, in_dim,  out_channels=1, y_dim=0, fp_dim=1024,device ='cpu', last_function='sigmoid'):
        super().__init__()
        self.z_dim = in_dim
        self.y_dim = y_dim
        self.device =device
        self.Linear1 = nn.Linear(in_dim + y_dim, 1024).to(self.device)
        self.Linear2 = nn.Linear(1024,1024).to(self.device)
        self.Linear3 = nn.Linear(in_dim, fp_dim).to(self.device)
        self.dropout1 = nn.Dropout(p=0.5)
        # self.Linear = nn.Linear(z_dim,fp_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose3d(16,16,3,padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(16,64,3,padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(16, out_channels, 3, padding=1),
            MetaActivation(last_function),
        )
        self.net = self.net.float()

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        h = self.Linear3(zy)
        h = torch.tanh(h)
        h = h.reshape(h.shape[0],16,4,4,4)
        return self.net(h)

#%%


class Deconv3D_DAT(nn.Module):
    """
    3 Dimensional Decoder architecture
    """
    def __init__(self, in_dim, out_channels=1, last_function='sigmoid'):
        self.in_dim = in_dim
        super(Deconv3D_DAT, self).__init__()
        self.fc_1 = nn.Linear(in_dim, in_dim)
        self.fc_2 = nn.Linear(in_dim, 2*in_dim)
        self.non_linearity = nn.Tanh()
        self.in_dim = in_dim
        self.ngf = 2
        self.layers = nn.ModuleList([
            nn.ConvTranspose3d(2*self.in_dim, 32 * self.ngf, 2, stride=2),
            nn.BatchNorm3d(32 * self.ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32*self.ngf, 16 * self.ngf, 2, stride=2),
            nn.BatchNorm3d(16 * self.ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16 * self.ngf, 8 * self.ngf, 2, stride=2),
            nn.BatchNorm3d(8 * self.ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(8 * self.ngf, 4 * self.ngf, 2, stride=2),
            nn.BatchNorm3d(4 * self.ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4 * self.ngf, 4 * self.ngf, 2, stride=2),
            nn.BatchNorm3d(4 * self.ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4 * self.ngf, 2 * self.ngf, 2, stride=2),
            nn.BatchNorm3d(2 * self.ngf),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(2 * self.ngf, out_channels, 2, stride=2),
            MetaActivation(last_function),
        ])
        print('Deconv decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        # 2 MLP Layers
        x = self.non_linearity(self.fc_1(x))
        x = self.fc_2(x)
        # Reshape
        x = x.view(-1, 2*self.in_dim, 1, 1, 1)
        # Deconvolutions
        for layer in self.layers:
            x = layer(x)

        return x[:,:,19:-18,9:-10,19:-18]