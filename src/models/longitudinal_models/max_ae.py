import torch
import torch.nn as nn
from src.models.networks.encoder import Convolutions_2D_64, MLP_variational, Encoding_1D, Convolutions_3D_64
from src.models.networks.decoder import Deconv2D_64, Decoder1D, Deconv3D_64

from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
from src.models.networks.permutation_factory import PermutationFactory
from src.models.networks.encoder import MLP_variational
from src.support.models_helper import reparametrize

class MaxAE(nn.Module):

    def __init__(self, data_info, latent_dimension, data_statistics, **kwargs):

        self.model_name = "max_ae"
        self.model_type = "max_ae"
        super(MaxAE, self).__init__()
        # TODO Paul : factoriser network_info ?
        network_info = {'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size' : kwargs["nn_size"],
                        'pi_module': False}


        latent_dimension_s = latent_dimension - 1
        latent_dimension_psi = 1
        self.latent_dimension_s = latent_dimension_s
        self.latent_dimension_psi = latent_dimension_psi
        self.latent_dimension = latent_dimension_s + latent_dimension_psi


        self.encoder = (EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension,
                                            network_info=network_info))[0]
        self.decoder = DecoderFactory.build(data_info=data_info, in_dim=self.latent_dimension,
                                            network_info=network_info)

        self.boost_mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=self.encoder.out_dim, out_features=32),
            #nn.LeakyReLU(),
            #nn.Linear(in_features=self.latent_dimension + 1, out_features=self.latent_dimension+1)
        )

        self.rnn = nn.RNN(input_size=33, hidden_size=32)

        self.mlp = MLP_variational(32, self.latent_dimension+1)

    def encode(self, obs):
        """
        observations -> latents
        """
        x = self.encoder(obs)
        x = x.reshape(x.shape[0], -1)
        return x

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def encode_time_space(self, observations, do_reparametrize=False, idx_pa=None, times=None):
        x = self.encode(observations)

        # Do the specific
        z_pa = []
        z_mu_list = []
        z_logvar_list = []
        for idx_pa, t in zip(idx_pa, times):
            z_temp = self.rnn(x[idx_pa].unsqueeze(1))[1].reshape(1,-1)
            z_mu, z_logvar = self.mlp(z_temp)
            if do_reparametrize:
                z_temp = reparametrize(z_mu, z_logvar)
            else:
                z_temp = z_mu
            psi_temp = torch.exp(z_temp[:,0])*(t-z_temp[:,1])
            zs_repeated = z_temp[:,2:].repeat(len(psi_temp),1)
            z_pa.append(torch.cat([psi_temp.reshape(-1,1), zs_repeated], axis=1))
            z_mu_list.append(z_mu)
            z_logvar_list.append(z_logvar)
        z = torch.cat(z_pa)
        return z


