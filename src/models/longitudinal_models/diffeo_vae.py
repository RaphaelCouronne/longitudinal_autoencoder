import numpy as np
from functools import reduce
from operator import mul
import torch.nn as nn
import torch
from src.models.networks.encoder import Convolutions_2D_64
from src.models.networks.decoder import Deconv2D_64

from src.models.networks.encoder import MLP_variational

from src.support.models_helper import reparametrize
from src.support.diffeo_helper import batched_vector_interpolation_adaptive
from src.support.images_helper import *

from src.models.networks.encoder_factory import EncoderFactory
from src.models.networks.decoder_factory import DecoderFactory
from src.models.networks.permutation_factory import PermutationFactory


class DVAE(nn.Module):
    """
    Semi-longitudinal Atlas model: Smooth deformations and Psi additional estimation
    (Static) Atlas is only learned at the zero disease stage, and is used as a basis for reconstruction
    """

    def __init__(self, data_info, latent_dimension, data_statistics, **kwargs):

        self.model_name = "referentialdiffeomorphic_vae"
        self.model_type = "diffeo"
        super(DVAE, self).__init__()

        # ----------- SET PARAMETERS
        initial_atlas = data_statistics[0].unsqueeze(0)
        self.decode_count = 0
        self.clamp_atlas = not kwargs['unclamp_atlas']
        self.tol = kwargs['tol']
        self.isometry_constraint = kwargs['isometry_constraint']
        self.nb_channels = initial_atlas.size(1)
        self.grid_size = tuple(initial_atlas.size()[2:])
        assert len(self.grid_size) in [2, 3], "Restricted to dimensions 2 and 3"

        # Latent space parameters
        self.latent_dimension = latent_dimension
        self.latent_dimension_s = latent_dimension - 1
        self.latent_dimension_psi = 1

        # Deformation parameters
        self.downsampling_grid = 2 ** kwargs['downsampling_grid']
        assert self.downsampling_grid in [1, 2, 4], "Only supports grid downsampling by 1, 2 and 4"
        self.downsampled_grid_size = tuple([gs // self.downsampling_grid for gs in self.grid_size])
        self.deformation_kernel_width = kwargs['deformation_kernel_width']

        # Set integration
        self.number_of_time_points = kwargs['number_of_time_points']
        self.dt = 1. / float(self.number_of_time_points - 1)
        # self.lambda_square = initial_lambda_square
        # self.mu_square = initial_mu_square
        self.noise_dimension = reduce(mul, self.grid_size)

        # Permutation invariance parameters
        self.pi_mode = kwargs['pi_mode']

        # ----------- SET NETWORKS
        # TODO Paul : factoriser network_info to incorporate decoder infos + permutation invariance mode ?
        network_info = {'decoder_grid_size': self.downsampled_grid_size,
                        'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size': kwargs["nn_size"],
                        'pi_module': True}


        self.space_encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension - 1, network_info=network_info)
        self.time_encoder = EncoderFactory.build(data_info=data_info, out_dim=1, network_info=network_info)
        self.decoder_s = DecoderFactory.build(data_info=data_info,
                                              network_info=network_info,
                                              in_dim=self.latent_dimension,
                                              out_channels=len(self.grid_size))
        self.encoder_out_dim = self.space_encoder[0].out_dim
        self.mlp_psi = MLP_variational(in_dim=self.encoder_out_dim, out_dim=1)
        self.pi_network = PermutationFactory.build(in_dim=self.encoder_out_dim,
                                                   out_dim=self.latent_dimension - 1,
                                                   mode=self.pi_mode)
        self.atlas = nn.Parameter(initial_atlas)

        print('>> Atlas intensities are {} = {} parameters'.format((self.atlas.size()[1:]), self.atlas.view(-1).size(0)))
        print('>> DVAE has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def vector_field_integration(self, v_):
        """
        vector field integration, akin to Transformer
        * scale and square algorithm
        outputs the diffeomorphism sampled as a warped grid
        """

        bts = v_.size(0)
        dim = len(self.grid_size)
        gs = self.grid_size
        dgs = self.downsampled_grid_size
        dsf = self.downsampling_grid  # (int) assumed identical along all dimensions
        ntp = self.number_of_time_points

        grid = torch.stack(torch.meshgrid([torch.linspace(0.0, elt - 1.0, delt) for elt, delt in zip(gs, dgs)])
                           ).type(str(v_.type())).view(*([1, dim] + list(dgs))).repeat(*([bts] + (dim + 1) * [1]))
        x = grid.clone() + v_ / float(2 ** ntp)

        # Scale & square integration scheme
        for t in range(1, self.number_of_time_points+1):
            x += self.dt * batched_vector_interpolation_adaptive(x - grid, x, dsf)
            assert not torch.isnan(x).any(), "NaN detected in grid during integration step"

        return x

    @staticmethod
    def apply_diffeomorphism(source, diffeomorphism):
        """
        Apply diffeomorphism to source (first)
        """
        assert not torch.isnan(source).any(), "NaN detected in source"
        warped_source = batched_scalar_interpolation_adaptive(source, diffeomorphism)
        assert not torch.isnan(warped_source).any(), "NaN detected in warped_source"
        return warped_source

    def encode(self, observations):
        """
        observations -> latents
        """
        pre_z_psi = self.time_encoder(observations)
        pre_z_s = self.space_encoder(observations)
        return pre_z_psi, pre_z_s

    def decode_s(self, z_psi, z):
        """
        latents -> field
        """

        bts = z.size(0)
        dim = len(self.grid_size)
        z_stacked = torch.cat((z_psi, z), dim=-1)

        # 1. Velocity field latent decoding
        latent_norm_squared = torch.sum(z.view(bts, -1) ** 2, dim=1)  # L2-norm
        v_star = self.decoder_s(z_stacked)  # momentum decoding
        v_ = batched_vector_smoothing(v_star, self.deformation_kernel_width, scaled=False)  # Kernel smoothing

        # 2. Forward normalization of velocity field | can enforce strict isometry (Z, l2) ~ (V, l_V)
        if self.isometry_constraint:
            v_norm_squared = torch.sum(v_ * v_star, dim=tuple(range(1, dim + 2)))
            normalizer = torch.where(latent_norm_squared > self.tol,
                                     torch.sqrt(latent_norm_squared / v_norm_squared),
                                     torch.from_numpy(np.array(0.0)).float().type(str(v_star.type())))
            normalizer = normalizer.view(*([latent_norm_squared.size(0)] + (dim + 1) * [1])).expand(v_.size())
            v_ = v_ * normalizer
        assert not torch.isnan(v_).any(), "NaN detected v"

        # 3. Displacement field generation (scaling and squaring)
        field = self.vector_field_integration(v_)
        return field

    def decode(self, z):
        x_hat = self.absolute_decode(z)
        return x_hat

    def absolute_decode(self, z):
        """
        (concatenated) latents --> reconstruction
        """
        z_psi, z_s = torch.split(z, split_size_or_sections=[self.latent_dimension_psi, self.latent_dimension_s], dim=-1)
        field_star = self.decode_s(z_psi, z_s)
        assert not torch.isnan(field_star).any(), "NaN detected field_star"
        assert not torch.isnan(z_psi).any(), "NaN detected z_psi"
        assert not torch.isnan(z_s).any(), "NaN detected z_s"

        # Reshaping (static) Atlas
        reshaped_atlas = self.atlas
        reshaped_atlas = reshaped_atlas.repeat(*([z_psi.shape[0]] + [1] * len(self.atlas.shape[1:])))
        if self.clamp_atlas:
            reshaped_atlas = torch.clamp(reshaped_atlas, self.tol, 1. - self.tol)

        # Reconstruction (by action of deformation)
        x_hat = self.apply_diffeomorphism(reshaped_atlas, field_star)
        return x_hat


class DRVAE(nn.Module):
    """
    Semi-longitudinal Atlas model: Smooth deformations and Psi additional estimation
    (Static) Atlas is only learned at the zero disease stage, and is used as a basis for reconstruction
    """

    def __init__(self, data_info, latent_dimension, data_statistics, **kwargs):

        self.model_name = "referentialdiffeomorphic_vae"    # same name (only relative aspect changes)
        self.model_type = "diffeo"
        super(DRVAE, self).__init__()

        # ----------- SET PARAMETERS
        initial_atlas = data_statistics[0].unsqueeze(0)
        self.decode_count = 0
        self.clamp_atlas = not kwargs['unclamp_atlas']
        self.tol = kwargs['tol']
        self.isometry_constraint = kwargs['isometry_constraint']
        self.nb_channels = initial_atlas.size(1)
        self.grid_size = tuple(initial_atlas.size()[2:])
        assert len(self.grid_size) in [2, 3], "Restricted to dimensions 2 and 3"

        # Latent space parameters
        self.latent_dimension = latent_dimension
        self.latent_dimension_s = latent_dimension - 1
        self.latent_dimension_psi = 1

        # Deformation parameters
        self.downsampling_grid = 2 ** kwargs['downsampling_grid']
        assert self.downsampling_grid in [1, 2, 4], "Only supports grid downsampling by 1, 2 and 4"
        self.downsampled_grid_size = tuple([gs // self.downsampling_grid for gs in self.grid_size])
        self.deformation_kernel_width = kwargs['deformation_kernel_width']

        # Set integration
        self.number_of_time_points = kwargs['number_of_time_points']
        self.dt = 1. / float(self.number_of_time_points - 1)
        # self.lambda_square = initial_lambda_square
        # self.mu_square = initial_mu_square
        self.noise_dimension = reduce(mul, self.grid_size)

        # Permutation invariance parameters
        self.pi_mode = kwargs['pi_mode']

        # ----------- SET NETWORKS
        # TODO Paul : factoriser network_info to incorporate decoder infos + permutation invariance mode ?
        network_info = {'decoder_grid_size': self.downsampled_grid_size,
                        'decoder_last_activation': kwargs['decoder_last_activation'],
                        'size': kwargs["nn_size"],
                        'pi_module': True}


        self.space_encoder = EncoderFactory.build(data_info=data_info, out_dim=self.latent_dimension - 1, network_info=network_info)
        self.time_encoder = EncoderFactory.build(data_info=data_info, out_dim=1, network_info=network_info)
        self.decoder_s = DecoderFactory.build(data_info=data_info,
                                              network_info=network_info,
                                              in_dim=self.latent_dimension,
                                              out_channels=len(self.grid_size))
        self.encoder_out_dim = self.space_encoder[0].out_dim
        self.mlp_psi = MLP_variational(in_dim=self.encoder_out_dim, out_dim=1)
        self.pi_network = PermutationFactory.build(in_dim=self.encoder_out_dim,
                                                   out_dim=self.latent_dimension - 1,
                                                   mode=self.pi_mode)
        self.atlas = nn.Parameter(initial_atlas)

        print('>> Atlas intensities are {} = {} parameters'.format((self.atlas.size()[1:]), self.atlas.view(-1).size(0)))
        print('>> DRVAE has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def vector_field_integration(self, v_):
        """
        vector field integration, akin to Transformer
        * scale and square algorithm
        outputs the diffeomorphism sampled as a warped grid
        """

        bts = v_.size(0)
        dim = len(self.grid_size)
        gs = self.grid_size
        dgs = self.downsampled_grid_size
        dsf = self.downsampling_grid  # (int) assumed identical along all dimensions
        ntp = self.number_of_time_points

        grid = torch.stack(torch.meshgrid([torch.linspace(0.0, elt - 1.0, delt) for elt, delt in zip(gs, dgs)])
                           ).type(str(v_.type())).view(*([1, dim] + list(dgs))).repeat(*([bts] + (dim + 1) * [1]))
        x = grid.clone() + v_ / float(2 ** ntp)

        # Scale & square integration scheme
        for t in range(1, self.number_of_time_points+1):
            x += self.dt * batched_vector_interpolation_adaptive(x - grid, x, dsf)
            assert not torch.isnan(x).any(), "NaN detected in grid during integration step"

        return x

    @staticmethod
    def apply_diffeomorphism(source, diffeomorphism):
        """
        Apply diffeomorphism to source (first)
        """
        assert not torch.isnan(source).any(), "NaN detected in source"
        warped_source = batched_scalar_interpolation_adaptive(source, diffeomorphism)
        assert not torch.isnan(warped_source).any(), "NaN detected in warped_source"
        return warped_source

    def atlas_anchors(self):
        # Specific Atlas anchoring | returns Atlas before permutation invariance module
        static_atlas_pre_latent_psi = self.time_encoder(self.atlas.detach())
        static_atlas_pre_latent_s = self.space_encoder(self.atlas.detach())
        return static_atlas_pre_latent_psi, static_atlas_pre_latent_s

    def atlas_anchoring(self, z_psi, z_s):
        static_atlas_latent_psi, static_atlas_latent_s = self.atlas_anchors()
        return z_psi - static_atlas_latent_psi, z_s - static_atlas_latent_s

    def encode(self, observations):
        """
        observations -> latents anchored
        """
        pre_z_psi = self.time_encoder(observations)
        pre_z_s = self.space_encoder(observations)
        pre_z_psi, pre_z_s = self.atlas_anchoring(pre_z_psi, pre_z_s)     # Perform Atlas Anchoring
        return pre_z_psi, pre_z_s

    def decode_s(self, z_psi, z):
        """
        latents -> field
        """

        bts = z.size(0)
        dim = len(self.grid_size)
        z_stacked = torch.cat((z_psi, z), dim=-1)

        # 1. Velocity field latent decoding
        latent_norm_squared = torch.sum(z.view(bts, -1) ** 2, dim=1)  # L2-norm
        v_star = self.decoder_s(z_stacked)  # momentum decoding
        v_ = batched_vector_smoothing(v_star, self.deformation_kernel_width, scaled=False)  # Kernel smoothing

        # 2. Forward normalization of velocity field | can enforce strict isometry (Z, l2) ~ (V, l_V)
        if self.isometry_constraint:
            v_norm_squared = torch.sum(v_ * v_star, dim=tuple(range(1, dim + 2)))
            normalizer = torch.where(latent_norm_squared > self.tol,
                                     torch.sqrt(latent_norm_squared / v_norm_squared),
                                     torch.from_numpy(np.array(0.0)).float().type(str(v_star.type())))
            normalizer = normalizer.view(*([latent_norm_squared.size(0)] + (dim + 1) * [1])).expand(v_.size())
            v_ = v_ * normalizer
        assert not torch.isnan(v_).any(), "NaN detected v"

        # 3. Displacement field generation (scaling and squaring)
        field = self.vector_field_integration(v_)
        return field

    def decode(self, z):
        x_hat = self.absolute_decode(z)
        return x_hat

    def absolute_decode(self, z):
        """
        (concatenated) latents --> reconstruction
        """
        z_psi, z_s = torch.split(z, split_size_or_sections=[self.latent_dimension_psi, self.latent_dimension_s], dim=-1)
        field_star = self.decode_s(z_psi, z_s)
        assert not torch.isnan(field_star).any(), "NaN detected field_star"
        assert not torch.isnan(z_psi).any(), "NaN detected z_psi"
        assert not torch.isnan(z_s).any(), "NaN detected z_s"

        # Reshaping (static) Atlas
        reshaped_atlas = self.atlas
        reshaped_atlas = reshaped_atlas.repeat(*([z_psi.shape[0]] + [1] * len(self.atlas.shape[1:])))
        if self.clamp_atlas:
            reshaped_atlas = torch.clamp(reshaped_atlas, self.tol, 1. - self.tol)

        # Reconstruction (by action of deformation)
        x_hat = self.apply_diffeomorphism(reshaped_atlas, field_star)
        return x_hat
