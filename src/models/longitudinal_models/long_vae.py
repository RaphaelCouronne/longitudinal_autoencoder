import torch
import torch.nn as nn
from src.models.networks.encoder import Convolutions_2D_64, MLP_variational, Encoding_1D, Convolutions_3D_64
from src.models.networks.decoder import Deconv2D_64, Decoder1D, Deconv3D_64

from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
from src.models.networks.permutation_factory import PermutationFactory
from src.models.networks.encoder import MLP_variational
from src.support.models_helper import reparametrize

class LongVAE(nn.Module):

    def __init__(self, data_info, latent_dimension, data_statistics, **kwargs):

        self.model_name = "longitudinal_vae"
        self.model_type = "agnostic"
        super(LongVAE, self).__init__()
        # TODO Paul : factoriser network_info ?
        network_info = {'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size': kwargs["nn_size"],
                        'pi_module': True}
        self.one_encoder = kwargs['one_encoder']

        latent_dimension_s = latent_dimension - 1
        latent_dimension_psi = 1
        self.latent_dimension_s = latent_dimension_s
        self.latent_dimension_psi = latent_dimension_psi
        self.latent_dimension = latent_dimension_s + latent_dimension_psi

        if self.one_encoder:
            self.encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info)
            self.encoder_out_dim = self.encoder[0].out_dim
        else:
            self.space_encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info)
            self.time_encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info)
            self.encoder_out_dim = self.space_encoder[0].out_dim


        self.mlp_psi = MLP_variational(in_dim=self.encoder_out_dim, out_dim=1)
        #self.mlp_s = MLP_variational(in_dim=64, out_dim=latent_dimension-1)

        #litmodel.model.mlp_psi = MLP_variational(in_dim=1, out_dim=1).cuda()
        #litmodel.model.mlp_s = MLP_variational(in_dim=latent_dimension - 1, out_dim=latent_dimension - 1).cuda()


        self.decoder = DecoderFactory.build(data_info=data_info, in_dim=self.latent_dimension,
                                            network_info=network_info)
        # Permutation invariance parameters
        self.pi_mode = kwargs['pi_mode']
        self.pi_network = PermutationFactory.build(in_dim=self.encoder_out_dim,
                                                   out_dim=self.latent_dimension - 1,
                                                   mode=self.pi_mode)
        #self.pi_network_logvar = PermutationFactory.build(dim=self.latent_dimension - 1, mode=self.pi_mode)

        # TODO Paul factorize
        #self.mu_square = kwargs['mu_square']


    """
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, observations):

        mean_psi = self.time_encoder[0](observations).reshape(-1, self.latent_dimension_psi)
        mean_s = self.space_encoder[0](observations).reshape(-1, self.latent_dimension_s)

        return mean_psi, 0, mean_s, 0

    def encode_space(self, observations, do_reparametrize=False):
        _, _, zs_mu, zs_logvar = self.encode(observations)
        return zs_mu

    def encode_time(self, observations, do_reparametrize=False):
        zpsi_mu, zpsi_logvar, _, _ = self.encode(observations)
        return zpsi_mu
    """


    def encode(self, observations):

        if self.one_encoder:
            pre_z = self.encoder(observations)
            pre_z_s, pre_z_psi = pre_z, pre_z
        else:
            pre_z_s = self.space_encoder(observations)
            #mean_s = self.space_encoder[0](observations)
            pre_z_psi = self.time_encoder(observations)
        return pre_z_psi, pre_z_s#, logvar_s

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



