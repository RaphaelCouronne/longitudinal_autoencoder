import torch
import torch.nn as nn
from src.models.networks.encoder import Convolutions_2D_64, MLP_variational, Encoding_1D, Convolutions_3D_64, Convolutions_2D_128, Convolutions_3D_64_Big, Convolutions_3D_DAT
from src.models.networks.decoder import Deconv2D_64, Decoder1D, Deconv3D_64


class EncoderFactory:

    @staticmethod
    def build(data_info, out_dim, network_info):
        dim = data_info["dim"]
        shape = data_info["shape"]
        size = network_info["size"]
        has_pi_module = network_info["pi_module"]
        modules = []
        if dim == 1:
            modules.append(Encoding_1D(in_dim=data_info["shape"], hidden_dim=32, out_dim=32))
        elif dim == 2:
            if shape == (64, 64):
                modules.append(Convolutions_2D_64())
            elif shape == (128, 128):
                modules.append(Convolutions_2D_128())
        elif dim == 3:
            if shape == (64, 64, 64):
                if size == "normal":
                    modules.append(Convolutions_3D_64())
                elif size == "big":
                    modules.append(Convolutions_3D_64_Big())
            elif shape == (128, 128, 128):
                raise NotImplementedError
            elif shape == (91, 109, 91):
                modules.append(Convolutions_3D_DAT())
        else:
            raise ValueError("Data dimension not handled")

        # Add mlp if no permutation invariance module
        if has_pi_module:
            # Flatten before pi_module
            modules.append(nn.Flatten())
        else:
            # MLP Variational
            modules.append(MLP_variational(in_dim=modules[0].out_dim, out_dim=out_dim))

        encoder = nn.Sequential(*modules)
        return encoder

        # TODO handle any dimension with interpolate ?
