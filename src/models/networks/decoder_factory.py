from src.models.networks.decoder import Deconv2D_64, Decoder1D, Deconv3D_64, Deconv2D_128, Deconv3D_64_big, Deconv3D_DAT
from src.models.networks.decoder_fields import DeconvBN2d_field, DeconvBN3d_field


class DecoderFactory:

    @staticmethod
    def build(data_info, in_dim, out_channels=1, network_info=None):

        # decoder arguments are network_info filtered by decoder_ str
        if network_info:
            decoder_args = {k: v for (k, v) in network_info.items() if 'decoder' in k}  # or startswith(decoder)
        else:
            decoder_args = None
            # raise ValueError("network_info should not be empty")

        # Additional info
        dim = data_info["dim"]
        shape = data_info["shape"]
        size = network_info["size"]
        last_function = decoder_args['decoder_last_activation'] if 'decoder_last_activation' else 'identity'  #TODO: factor
        is_field = 'decoder_grid_size' in decoder_args  # only mention grid size reduction for velocity field decoders
        assert (out_channels == 1) or ((out_channels > 1) and dim in [2, 3]), "Output channels errors"

        if dim == 1:
            decoder = Decoder1D(in_dim=in_dim, out_dim=shape, last_function=last_function)
        elif dim == 2:
            if is_field:
                decoder = DeconvBN2d_field(in_dim=in_dim, out_grid_size=decoder_args['decoder_grid_size'],
                                           out_channels=out_channels, final_activation=last_function)
            else:
                if shape == (64, 64):
                    decoder = Deconv2D_64(in_dim=in_dim, out_channels=out_channels,
                                          last_function=last_function)
                elif shape == (128, 128):
                    decoder = Deconv2D_128(in_dim=in_dim, out_channels=out_channels,
                                           last_function=last_function)
        elif dim == 3:
            if is_field:
                decoder = DeconvBN3d_field(in_dim=in_dim, out_grid_size=decoder_args['decoder_grid_size'],
                                           out_channels=out_channels, final_activation=last_function)
            else:
                if shape == (64, 64, 64):
                    if size == "normal":
                        decoder = Deconv3D_64(in_dim=in_dim, out_channels=out_channels,
                                              last_function=last_function)
                    elif size == "big":
                        decoder = Deconv3D_64_big(in_dim=in_dim, out_channels=out_channels,
                                              last_function=last_function)
                    else:
                        raise NotImplementedError

                elif shape == (128, 128, 128):
                    raise NotImplementedError
                elif shape == (91, 109, 91):
                    decoder = Deconv3D_DAT(in_dim=in_dim, out_channels=out_channels,
                                          last_function=last_function)
        else:
            raise ValueError("Data dimension not handled")
        return decoder

        # TODO handle any dimension with interpolate ?
