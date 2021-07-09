import torch
import torch.nn as nn
from src.models.networks.encoder import Convolutions_2D_64, MLP_variational, Encoding_1D, Convolutions_3D_64
from src.models.networks.decoder import Deconv2D_64, Decoder1D, Deconv3D_64

from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
from src.models.networks.permutation_factory import PermutationFactory

from src.support.models_helper import reparametrize

class MLVAE(nn.Module):

    def __init__(self, data_info, latent_dimension, data_statistics, **kwargs):

        self.model_name = "mlvae"
        self.model_type = "mlvae"
        super(MLVAE, self).__init__()
        # TODO Paul : factoriser network_info ?
        network_info = {'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size' : kwargs["nn_size"],
                        'pi_module': False}
        self.one_encoder = kwargs['one_encoder']

        latent_dimension_s = latent_dimension - 1
        latent_dimension_psi = 1
        self.latent_dimension_s = latent_dimension_s
        self.latent_dimension_psi = latent_dimension_psi
        self.latent_dimension = latent_dimension_s + latent_dimension_psi

        if self.one_encoder:
            self.encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info)
        else:
            self.space_encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension-1,
                                            network_info=network_info)
            self.time_encoder = EncoderFactory.build(data_info=data_info, out_dim=1,
                                            network_info=network_info)

        self.decoder = DecoderFactory.build(data_info=data_info, in_dim=self.latent_dimension,
                                            network_info=network_info)

        # TODO Paul factorize
        #self.mu_square = kwargs['mu_square']

    def encode(self, observations):
        """
        observations -> latents
        """
        if self.one_encoder:
            mean, logvar = self.encoder(observations)
            mean_s, mean_psi = mean[:,:self.latent_dimension_s], mean[:,self.latent_dimension_s:]
            logvar_s, logvar_psi = logvar[:,:self.latent_dimension_s], logvar[:,self.latent_dimension_s:]
        else:
            mean_s, logvar_s = self.space_encoder(observations)
            mean_psi, logvar_psi = self.time_encoder(observations)
        return mean_psi, logvar_psi, mean_s, logvar_s

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def encode_space(self, observations, do_reparametrize=False):
        _, _, zs_mu, zs_logvar = self.encode(observations)
        if do_reparametrize:
            zs = reparametrize(zs_mu, zs_logvar)
            return zs
        else:
            return zs_mu

    def encode_time(self, observations, do_reparametrize=False):
        zpsi_mu, zpsi_logvar, _, _ = self.encode(observations)
        if do_reparametrize:
            zpsi = reparametrize(zpsi_mu, zpsi_logvar)
            return zpsi
        else:
            return zpsi_mu


