import torch
import torch.nn as nn
from src.models.networks.permutation import IdentityPermutation, MeanPermutation, MaxPermutation, DeepPermutation, RNNPermutation, DeepPermutationSimple


class PermutationFactory:

    @staticmethod
    def build(in_dim, out_dim, hidden_dim=None, mode=None):
        if mode == "identity":
            pi_network = IdentityPermutation(in_dim, out_dim, hidden_dim)
        #elif mode == 'mean':
        #    pi_network = MeanPermutation()
        #elif mode == 'max':
        #    pi_network = MaxPermutation()
        elif mode in ["mean", "max"]:
            pi_network = DeepPermutationSimple(in_dim, out_dim, hidden_dim, operator=mode)
        elif mode == 'RNN':
            raise ValueError("RNN not implemented")
            pi_network = RNNPermutation(in_dim)
        else:
            raise NotImplementedError
        return pi_network
