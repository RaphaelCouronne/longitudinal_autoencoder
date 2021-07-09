import torch.nn as nn
from src.models.networks.encoder import Convolutions_2D_64, MLP_variational, Encoding_1D
from src.models.networks.decoder import Deconv2D_64, Decoder1D
from src.support.models_helper import reparametrize
from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
import torch
from src.support.models_helper import reparametrize


class UnitNormLinear(nn.Linear):
    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight.div(self.weight.norm()), self.bias)


class MaxNormLinear(nn.Linear):
    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight.clamp(max=0), self.bias)


class BVAE_Regression(nn.Module):

    def __init__(self, data_info, latent_dimension, data_statistics,**kwargs):
        self.model_name = "BVAE_Regression"
        self.model_type = "vae_regr"
        super(BVAE_Regression, self).__init__()
        # TODO Paul : factoriser network_info ?
        network_info = {'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size' : kwargs["nn_size"],
                        'pi_module': False}
        latent_dimension_s = latent_dimension - 1
        latent_dimension_t = 1
        self.latent_dimension = latent_dimension
        self.latent_dimension_s = latent_dimension_s
        self.latent_dimension_t = latent_dimension_t

        self.encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info)
        self.decoder = DecoderFactory.build(data_info=data_info, in_dim=self.latent_dimension,
                                            network_info=network_info)

        self.linear_r_mu = torch.nn.Linear(in_features=self.latent_dimension, out_features=1)
        self.linear_r_logvar = torch.nn.Linear(in_features=self.latent_dimension, out_features=1)
        self.u = UnitNormLinear(in_features=1, out_features=self.latent_dimension, bias=False)
        self.u_var = MaxNormLinear(in_features=1, out_features=1, bias=False)

    def encode(self, obs):
        z_mu, z_logvar = self.encoder(obs) #TODO handle also linear for z_mu
        r_mu = self.linear_r_mu(torch.tanh(z_mu))
        r_logvar = self.linear_r_logvar(torch.tanh(z_mu))
        return z_mu, z_logvar, r_mu, r_logvar

    def decode(self, x):
        return self.decoder(x)


    def encode_space(self, observations, do_reparametrize=False):
        mean_zs_all, logvar_zs_all, mean_zpsi, logvar_zpsi = self.encode(observations)
        if do_reparametrize:
            zs = reparametrize(mean_zs_all, logvar_zs_all)
            zpsi = reparametrize(mean_zpsi, logvar_zpsi)
        else:
            zs = mean_zs_all
        mean_zs_all = zs - self.u(zpsi)
        return mean_zs_all

    def encode_time(self, observations, do_reparametrize=False):
        _, _, mean_zpsi, logvar_zpsi = self.encode(observations)
        if do_reparametrize:
            zpsi = reparametrize(mean_zpsi, logvar_zpsi)
        else:
            zpsi = mean_zpsi
        return zpsi

# TODO other methods for linear layer normalization

#from torch.nn.utils import weight_norm as wn
#time_layer = wn(nn.Linear(self.latent_dimension, 1, bias=False).cuda())

# class UnitNorm():
#     """
#     UnitNorm constraint.
#     Constraints the weights to have column-wise unit norm
#     """
#     def __init__(self,
#                  frequency=1,
#                  unit='batch',
#                  module_filter='*'):
#
#         self.frequency = frequency
#         self.unit = unit
#         self.module_filter = module_filter
#
#     def __call__(self, module):
#         w = module.weight.data
#         module.weight.data = w.div(torch.linalg.norm(w))
#
# import torch.functional as F

# https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
# https://discuss.pytorch.org/t/kernel-constraint-similar-to-the-one-implemented-in-keras/49936

