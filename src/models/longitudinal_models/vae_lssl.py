import torch.nn as nn
from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
import torch
from src.models.networks.encoder import MLP_variational
from src.support.models_helper import reparametrize

class VaeLSSL(nn.Module):

    def __init__(self, data_info, latent_dimension, data_statistics, **kwargs):
        self.model_name = "vae_lssl"
        self.model_type = "vae_lssl"
        super(VaeLSSL, self).__init__()
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
        # Time direction
        init_u = torch.zeros(size=(1, self.latent_dimension))
        init_u[0][0] = 1
        self.u = torch.nn.Parameter(init_u)

    def encode(self, obs):
         z_mu, z_logvar = self.encoder(obs) #TODO handle also linear for z_mu
         return z_mu, z_logvar

    def decode(self, x):
        return self.decoder(x)

    def encode_space(self, observations, do_reparametrize=False):
        z_mu, z_logvar, = self.encode(observations)
        if do_reparametrize:
            z = reparametrize(z_mu, z_logvar)
        else:
            z = z_mu
        mean_psi_all = torch.matmul(z, self.u.T)
        mean_zs_all = z - torch.matmul(mean_psi_all, self.u)
        return mean_zs_all

    def encode_time(self, observations, do_reparametrize=False):
        z_mu, z_logvar, = self.encode(observations)
        if do_reparametrize:
            z = reparametrize(z_mu, z_logvar)
        else:
            z = z_mu
        mean_psi_all = torch.matmul(z, self.u.T)
        return mean_psi_all
