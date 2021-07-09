import torch.nn as nn
from src.models.networks.encoder import Convolutions_2D_64, MLP_variational, Encoding_1D
from src.models.networks.decoder import Deconv2D_64, Decoder1D
from src.support.models_helper import reparametrize
from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
import torch

class BVAE(nn.Module):

    def __init__(self, data_info, latent_dimension, data_statistics,**kwargs):
        self.model_name = "BVAE"
        self.model_type = "vae"
        super(BVAE, self).__init__()
        # TODO Paul : factoriser network_info ?
        network_info = {'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size' : kwargs["nn_size"],
                        'pi_module': False}
        latent_dimension_s = latent_dimension - 1
        latent_dimension_psi = 1
        self.latent_dimension = latent_dimension
        self.latent_dimension_s = latent_dimension_s
        self.latent_dimension_psi = latent_dimension_psi

        self.encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info)
        self.decoder = DecoderFactory.build(data_info=data_info, in_dim=self.latent_dimension,
                                            network_info=network_info)

    def encode(self, obs):
        z_mu, z_logvar = self.encoder(obs)
        return z_mu, z_logvar#return z_mu[:,1:], z_logvar[:,1:], z_mu[:,0].unsqueeze(1), z_logvar[:,0].unsqueeze(1)

    def decode(self, x):
        return self.decoder(x)
