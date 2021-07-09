import torch
import torch.nn as nn
from src.models.networks import MetaActivation
from functools import reduce
from operator import mul


class IdentityPermutation(nn.Module):
    """
    No permutation invariance : identity function
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, nonlinearity="lrelu", *args, **kwargs):
        nn.Module.__init__(self)

        if hidden_dim is None:
            hidden_dim = in_dim

        print('>> IdentityPermutation has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

        self.rho_mu = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 MetaActivation(nonlinearity),
                                 #nn.Linear(32, 32),
                                 #MetaActivation(nonlinearity),
                                 nn.Linear(hidden_dim, out_dim)
                                 )

        self.rho_sigma = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 MetaActivation(nonlinearity),
                                 #nn.Linear(hidden_dim, hidden_dim),
                                 #MetaActivation(nonlinearity),
                                 nn.Linear(hidden_dim, out_dim)
                                 )


    def forward(self, x):
        return self.rho_mu(x), self.rho_sigma(x)


class MeanPermutation(nn.Module):
    """
    Permutation invariance by (dimension-wise) mean (ie barycenter)
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        print('>> MeanPermutation has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        assert len(x.shape) == 3, "format x to be (bts, visits, latent dims)"
        res = torch.mean(x, dim=1, keepdim=True)
        return res


class MaxPermutation(nn.Module):
    """
    Permutation invariance by (dimension-wise) max
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        print('>> MaxPermutation has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        assert len(x.shape) == 3, "format x to be (bts, visits, latent dims)"
        res = torch.max(x, dim=1, keepdim=True)[0]
        return res


class DeepPermutation(nn.Module):
    """
    Permutation invariance according to Deep Set strategy
    """

    def __init__(self, dim, nonlinearity='relu'):
        nn.Module.__init__(self)
        self.dim = dim
        self.phi = nn.Sequential(nn.Linear(self.dim, 2 * self.dim),
                                 MetaActivation(nonlinearity),
                                 nn.Linear(2 * self.dim, 2 * self.dim),
                                 MetaActivation(nonlinearity)
                                 )
        self.rho = nn.Sequential(nn.Linear(2 * self.dim, 2 * self.dim),
                                 MetaActivation(nonlinearity),
                                 nn.Linear(2 * self.dim, 2 * self.dim),
                                 MetaActivation(nonlinearity),
                                 nn.Linear(2 * self.dim, self.dim)
                                 )
        print('>> DeepPermutation has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        assert len(x.shape) == 3, "format x to be (bts, visits, latent dims)"
        bts, nb_visits, latent_dim = x.shape
        invariant_representation = torch.mean(self.phi(x.reshape(bts * nb_visits, latent_dim)
                                                       ).reshape(bts, nb_visits, -1),
                                              dim=1, keepdim=False)
        res = self.rho(invariant_representation).unsqueeze(1)  # (re)-add visit dimension at 1
        return res


class DeepPermutationSimple(nn.Module):
    """
    Permutation invariance according to Deep Set strategy
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, nonlinearity='lrelu', operator="mean"):
        nn.Module.__init__(self)
        assert operator in ["mean", "max"]
        self.operator = operator
        self.in_dim = in_dim
        self.out_dim = out_dim
        if hidden_dim is None:
            hidden_dim = in_dim
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(#MetaActivation(nonlinearity),
                                 #nn.Linear(self.dim, self.dim),
                                 #MetaActivation(nonlinearity),
                                 #nn.Linear(hidden_dim, hidden_dim),
                                 )
        self.rho_mu = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 MetaActivation(nonlinearity),
                                 #nn.Linear(32, 32),
                                 #MetaActivation(nonlinearity),
                                 nn.Linear(hidden_dim, out_dim)
                                 )

        self.rho_sigma = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 MetaActivation(nonlinearity),
                                 #nn.Linear(hidden_dim, hidden_dim),
                                 #MetaActivation(nonlinearity),
                                 nn.Linear(hidden_dim, out_dim)
                                 )

        print('>> DeepPermutation has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        assert len(x.shape) == 3, "format x to be (bts, visits, latent dims)"
        bts, nb_visits, latent_dim = x.shape
        #invariant_representation = torch.mean(x, dim=1, keepdim=False)

        phi_representation = self.phi(x.reshape(bts * nb_visits, latent_dim)).reshape(bts, nb_visits, -1)
        if self.operator == "mean":
            invariant_representation = torch.mean(phi_representation,dim=1, keepdim=False)
        elif self.operator == "max":
            invariant_representation = torch.max(phi_representation,dim=1, keepdim=False)[0]
        else:
            raise NotImplementedError
        z_mu = self.rho_mu(invariant_representation).unsqueeze(1)  # (re)-add visit dimension at 1
        z_sigma = self.rho_sigma(invariant_representation).unsqueeze(1)
        #res = invariant_representation.unsqueeze(1)
        return z_mu, z_sigma

class RNNPermutation(nn.Module):
    """
    RNN dummy class
    """

    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        # TODO: Add simple RNN network for concatenation
        print('>> RNNPermutation has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))
        pass

    def forward(self, x):
        assert len(x.shape) == 3, "format x to be (bts, visits, latent dims)"
        pass

