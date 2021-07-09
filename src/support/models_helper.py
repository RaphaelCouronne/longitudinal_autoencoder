import torch
import numpy as np
import sys
sys.path.append("lib/fast-soft-sort")
from fast_soft_sort.pytorch_ops import soft_rank
from src.models.networks.permutation_factory import IdentityPermutation

def reparametrize(mean, logvariance):
    std = torch.exp(0.5 * logvariance)
    return mean + torch.zeros_like(std).normal_() * std

def get_modelrelated_hparams(model_name):
    assert model_name in ['BVAE', 'LongVAE', 'DVAE', "BVAE_Regression", "VaeLSSL", "MaxVAE", "MLVAE", "DRVAE", "MaxAE"], "Model name not found"
    # Common keys
    list_str = ['decoder_last_activation', 'pi_mode', 'nn_size']
    # model related keys
    if model_name in ['DVAE',"DRVAE"]:
        list_str += ['number_of_time_points', 'downsampling_grid',
                     'deformation_kernel_width', 'unclamp_atlas', 'isometry_constraint', 'tol']
    elif model_name == 'LongVAE':
        list_str+=["one_encoder"]
    elif model_name == 'MLVAE':
        list_str+=["one_encoder"]
    return list_str

def moving_averager(alpha_smoothing, current, previous, is_first=False):
    """Moving averaging for GECO smoothing"""
    return current if is_first else alpha_smoothing * previous + (1 - alpha_smoothing) * current

def gpu_numpy_detach(x):
    return x.cpu().detach().numpy()

#####################
###### Losses #######
#####################

def get_attachment_loss(name):
    assert name in ['mse'], "Attachment loss not recognized"
    att_loss = torch.nn.modules.loss.MSELoss(**{'reduction': 'sum'})
    return att_loss

def compute_kl(z_mu, z_logvar, type, prior_mu=None, prior_logvar=None):
    if prior_logvar is None:
        prior_logvar = torch.Tensor([0]).type(type)
    if prior_mu is None:
        prior_mu = torch.Tensor([0]).type(type)

    log_part = z_logvar - prior_logvar
    ratio_var = torch.exp(z_logvar) / torch.exp(prior_logvar)
    diff_mu = ((prior_mu - z_mu) ** 2) / torch.exp(prior_logvar)
    regularity_loss = 0.5 * torch.sum(-log_part + ratio_var + diff_mu - 1)
    return regularity_loss


def compute_soft_spearman(litmodel, indices_1, b1_times, b1_z_psi, nb_visits, nb_patients, evaluate=False):
    # TODO fix for now when number of visits differs : set to 3
    if nb_visits == 0 or evaluate:
        nb_visits_spearman = 3
        indices_spearman = [
            sorted(litmodel.r.choice(len(idx_1), size=3, replace=len(idx_1) < 3))
            for idx_1 in indices_1]

        b1_times_spearman = torch.cat([b1_times[idx_1][idx_spearman] for idx_1, idx_spearman in zip(indices_1, indices_spearman)])
        b1_z_psi_spearman = torch.cat([b1_z_psi[idx_1][idx_spearman] for idx_1, idx_spearman in zip(indices_1, indices_spearman)])

        return compute_soft_spearman_batch(litmodel, indices_spearman, b1_times_spearman, b1_z_psi_spearman, nb_visits_spearman, nb_patients)
    else:
        return compute_soft_spearman_batch(litmodel, indices_1, b1_times, b1_z_psi, nb_visits, nb_patients)

def compute_soft_spearman_batch(litmodel, indices_1, b1_times, b1_z_psi, nb_visits, nb_patients):

    # Spearman loss
    ranking_loss = torch.Tensor([0])
    ranking_loss_w = 1.
    if litmodel.on_gpu:
        ranking_loss = ranking_loss.cuda(litmodel.last_device)
    if litmodel.softrank['use']:
        for i, e in enumerate(indices_1):
            # Observe unique indices of visits drawn randomly
            _, indices_u = np.unique(e, return_index=True)
            if len(indices_u) > 1:
                # enough indices to perform ranking for this subject
                mask_unique_indices = [i in indices_u for i in range(nb_visits)]  # boolean mask
                true_soft_rank_i = soft_rank(b1_times.view(*[nb_patients,
                                                             nb_visits,
                                                             1])[i][mask_unique_indices].cpu().view(1, -1),
                                             regularization_strength=litmodel.softrank['reg']
                                             ).float()
                pred_soft_rank_i = soft_rank(b1_z_psi.view(*[nb_patients,
                                                             nb_visits,
                                                             litmodel.model.latent_dimension_psi])[i][
                                                 mask_unique_indices].view(1, -1).cpu(),
                                             regularization_strength=litmodel.softrank['reg']
                                             ).float()
                ranking_loss += torch.sum((pred_soft_rank_i - true_soft_rank_i) ** 2)
                ranking_loss_w += len(true_soft_rank_i)
            else:
                # not enough visits for ranking, pass to the next subject (<==> we put a weight 0. to this tanking)
                pass
    ranking_loss /= ranking_loss_w
    return ranking_loss

####################################
###### Permuation Invariance #######
####################################

def get_indices(litmodel, batch, nb_space_visits, evaluate=False):
    observations_list = batch['obs']

    if litmodel.random_select > 0:

        # Batch n°1 (b1): target individuals to be reconstructed
        indices_1 = [sorted(litmodel.r.choice(len(obs), size=litmodel.random_select, replace=len(obs) < litmodel.random_select))
                     for obs in observations_list]

        # Batch n°2 (b2): helper individuals to generate latent space shift code
        indices_2 = [litmodel.r.choice(len(obs), size=nb_space_visits, replace=True) for obs in
                     observations_list]

    else:
        indices = [list(np.array(x)-x[0]) for x in batch["idx_pa"]]
        indices_1, indices_2 = indices, indices

    if evaluate:
        indices_1 = [list(np.array(x) - x[0]) for x in batch["idx_pa"]]

    return indices_1, indices_2


def get_latent_perm_invariance(litmodel, indices_1, indices_2, dates, obs_list, nb_patients, nb_visits, nb_space_visits, evaluate=False):
    """
    Latent permutation invariance & forward encoding
    indices_1: (list of list) longitudinal indices per subject in the batch
    indices_2: (list of list) spatial indices per subject in the batch
    """
    # TODO Paul : weird here that we make more computations than we should

    #TODO if subsample & identity : indices2 has to match indices1
    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        indices_2 = indices_1
        nb_space_visits = nb_visits

    # Stack representations
    nested_t_1 = [torch.stack([t[i] for i in ind], 0) for ind, t in zip(indices_1, dates)]
    b1_obs = torch.cat([obs[ind_1] for obs, ind_1 in zip(obs_list, indices_1)])
    b1_times = torch.cat(nested_t_1)
    b2_obs = torch.cat([obs[ind_2] for obs, ind_2 in zip(obs_list, indices_2)])
    bfull_obs = torch.cat([b1_obs, b2_obs])

    # A. FORWARD | Common forward pass on stacked inputs
    pre_z_psi, pre_z_s = litmodel.model.encode(bfull_obs)
    z_psi_mu, z_psi_logvar = litmodel.model.mlp_psi(pre_z_psi)

    #z_s = z_s.reshape(z_s.shape[0], -1)
    # Asserts
    assert not torch.isnan(bfull_obs).any(), "NaN detected bfullobs"
    litmodel.last_device = b1_obs.device.index
    assert litmodel.last_device == b1_obs.device.index, "Dates and observations are not on the same device !"
    assert not torch.isnan(z_psi_mu).any(), "NaN detected encoding"

    # Psi sample
    #b1_z_psi_mu, b1_z_psi_logvar = litmodel.model.mlp_psi(z_psi)
    z_psi_logvar = torch.clamp(z_psi_logvar, litmodel.hparams.cliplogvar_min,
                                   litmodel.hparams.cliplogvar_max)
    b1_z_psi_mu, b1_z_psi_logvar = z_psi_mu, z_psi_logvar
    rb1_z_psi_mu = b1_z_psi_mu
    z_psi_sampled = reparametrize(z_psi_mu, z_psi_logvar)

    # Space sample
    #z_s_sampled = reparametrize(z_s, z_s_logvar)

    # Cut per group of subsampling
    b1_z_s, b2_z_s = torch.split(pre_z_s, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)
    #b1_z_s_logvar, b2_z_s_logvar = torch.split(z_s_logvar, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)
    b1_z_psi, b2_z_psi = torch.split(z_psi_sampled, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)
    rb1_z_psi_mu, _ = torch.split(z_psi_mu, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)


    # If identity module
    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        # Call permutation invariance network
        b2_z_s_PI_mu, b2_z_s_PI_logvar = litmodel.model.pi_network(b2_z_s)
        # Clip
        b2_z_s_PI_logvar = torch.clamp(b2_z_s_PI_logvar, litmodel.hparams.cliplogvar_min,
                                       litmodel.hparams.cliplogvar_max)
        # Sample
        rb1_z_s_PI = reparametrize(b2_z_s_PI_mu, b2_z_s_PI_logvar)
        rb1_z_s_PI_mu = b2_z_s_PI_mu

    # If can be batched
    elif nb_visits > 0 and (not evaluate):
        # Reshape
        #b2_z_s_view = b2_z_s.view(*[nb_patients, nb_space_visits, litmodel.model.latent_dimension_s])
        b2_z_s_view = b2_z_s.view(*[nb_patients, nb_space_visits, pre_z_s.shape[1]])
        #b2_z_s_view_logvar = b2_z_s_logvar.view(*[nb_patients, nb_space_visits, z_s_logvar.shape[1]])
        # Call permutation invariance network
        b2_z_s_PI_mu, b2_z_s_PI_logvar = litmodel.model.pi_network(b2_z_s_view)
        # Clip
        b2_z_s_PI_logvar = torch.clamp(b2_z_s_PI_logvar, litmodel.hparams.cliplogvar_min, litmodel.hparams.cliplogvar_max)

        # Expand
        rb1_z_s_PI_mu = b2_z_s_PI_mu.repeat(*([1, nb_visits, 1])).reshape(nb_patients * nb_visits,
                                                                    litmodel.model.latent_dimension_s)
        rb1_z_s_PI_logvar = b2_z_s_PI_logvar.repeat(*([1, nb_visits, 1])).reshape(nb_patients * nb_visits,
                                                                    litmodel.model.latent_dimension_s)
        # Sample
        rb1_z_s_PI = reparametrize(rb1_z_s_PI_mu, rb1_z_s_PI_logvar)

    # If cannot be batched
    elif nb_visits == 0 or evaluate:
        # Get positions for split per patient
        lengths = [0] + [len(x) for x in indices_2]
        positions = np.add.accumulate(lengths)
        positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]
        # Call permutation invariance network
        b2_z_s_PI_mu_list = []
        b2_z_s_PI_logvar_list = []
        for pos in positions:
            b2_z_s_PI_mu_temp, b2_z_s_PI_logvartemp = litmodel.model.pi_network(b2_z_s[pos].unsqueeze(0))
            b2_z_s_PI_mu_list.append(b2_z_s_PI_mu_temp)
            b2_z_s_PI_logvar_list.append(b2_z_s_PI_logvartemp)

        b2_z_s_PI_mu, b2_z_s_PI_logvar = torch.cat(b2_z_s_PI_mu_list), torch.cat(b2_z_s_PI_logvar_list)
        #b2_z_s_PI_logvar = torch.cat([litmodel.model.pi_network_logvar(b2_z_s_logvar[pos].unsqueeze(0)) for pos in positions])
        # Clip
        b2_z_s_PI_logvar = torch.clamp(b2_z_s_PI_logvar, litmodel.hparams.cliplogvar_min,
                                       litmodel.hparams.cliplogvar_max)
        # Expand
        rb1_z_s_PI_mu = torch.cat([b2_z_s_PI_mu[i].repeat(len(idx_1), 1) for i, idx_1 in enumerate(indices_1)], axis=0)
        rb1_z_s_PI_logvar = torch.cat([b2_z_s_PI_logvar[i].repeat(len(idx_1), 1) for i, idx_1 in enumerate(indices_1)],axis=0)
        # Sample
        rb1_z_s_PI = reparametrize(rb1_z_s_PI_mu, rb1_z_s_PI_logvar)

    return b1_z_psi, rb1_z_s_PI, \
           b1_obs, b1_times, \
           b2_z_s_PI_mu, b2_z_s_PI_logvar, \
           b1_z_psi_mu, b1_z_psi_logvar, \
           rb1_z_psi_mu, rb1_z_s_PI_mu
           # add the mu for metrics




def get_latent_perm_invariance_oldus(litmodel, indices_1, indices_2, dates, obs_list, nb_patients, nb_visits, nb_space_visits, evaluate=False):
    """
    Latent permutation invariance & forward encoding
    indices_1: (list of list) longitudinal indices per subject in the batch
    indices_2: (list of list) spatial indices per subject in the batch
    """
    # TODO Paul : weird here that we make more computations than we should

    #TODO if subsample & identity : indices2 has to match indices1
    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        indices_2 = indices_1
        nb_space_visits = nb_visits

    # Stack representations
    nested_t_1 = [torch.stack([t[i] for i in ind], 0) for ind, t in zip(indices_1, dates)]
    b1_obs = torch.cat([obs[ind_1] for obs, ind_1 in zip(obs_list, indices_1)])
    b1_times = torch.cat(nested_t_1)
    b2_obs = torch.cat([obs[ind_2] for obs, ind_2 in zip(obs_list, indices_2)])
    bfull_obs = torch.cat([b1_obs, b2_obs])

    # A. FORWARD | Common forward pass on stacked inputs
    z_psi, z_psi_logvar, z_s_mu, z_s_logvar = litmodel.model.encode(bfull_obs)
    #z_s = z_s.reshape(z_s.shape[0], -1)
    # Asserts
    assert not torch.isnan(bfull_obs).any(), "NaN detected bfullobs"
    litmodel.last_device = b1_obs.device.index
    assert litmodel.last_device == b1_obs.device.index, "Dates and observations are not on the same device !"
    assert not torch.isnan(z_psi).any(), "NaN detected encoding"

    # Psi sample
    #b1_z_psi_mu, b1_z_psi_logvar = litmodel.model.mlp_psi(z_psi)
    z_psi_logvar = torch.clamp(z_psi_logvar, litmodel.hparams.cliplogvar_min,
                                   litmodel.hparams.cliplogvar_max)
    b1_z_psi_mu, b1_z_psi_logvar = z_psi, z_psi_logvar
    z_psi_sampled = reparametrize(z_psi, z_psi_logvar)

    # Space sample
    #z_s_sampled = reparametrize(z_s, z_s_logvar)

    # Cut per group of subsampling
    b1_z_s_mu, b2_z_s_mu = torch.split(z_s_mu, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)
    b1_z_s_logvar, b2_z_s_logvar = torch.split(z_s_logvar, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)
    b1_z_psi, b2_z_psi = torch.split(z_psi_sampled, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)

    # If identity module
    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        lengths = [0] + [len(x) for x in indices_2]
        positions = np.add.accumulate(lengths)
        positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]
        b2_z_s_PI_mu = torch.cat([litmodel.model.pi_network(b2_z_s_mu[pos]) for pos in positions])
        b2_z_s_PI_logvar = torch.cat([litmodel.model.pi_network_logvar(b2_z_s_logvar[pos]) for pos in positions])
        # Clip
        b2_z_s_PI_logvar = torch.clamp(b2_z_s_PI_logvar, litmodel.hparams.cliplogvar_min,
                                       litmodel.hparams.cliplogvar_max)
        # Sample
        rb1_z_s_PI = reparametrize(b2_z_s_PI_mu, b2_z_s_PI_logvar)

    # If can be batched
    elif nb_visits > 0 and (not evaluate):
        # Reshape
        #b2_z_s_view = b2_z_s.view(*[nb_patients, nb_space_visits, litmodel.model.latent_dimension_s])
        b2_z_s_view_mu = b2_z_s_mu.view(*[nb_patients, nb_space_visits, z_s_mu.shape[1]])
        b2_z_s_view_logvar = b2_z_s_logvar.view(*[nb_patients, nb_space_visits, z_s_logvar.shape[1]])
        # Call permutation invariance network
        b2_z_s_PI_mu = litmodel.model.pi_network(b2_z_s_view_mu)
        b2_z_s_PI_logvar = litmodel.model.pi_network_logvar(b2_z_s_view_logvar)
        # Clip
        b2_z_s_PI_logvar = torch.clamp(b2_z_s_PI_logvar, litmodel.hparams.cliplogvar_min, litmodel.hparams.cliplogvar_max)

        # Expand
        rb1_z_s_PI_mu = b2_z_s_PI_mu.repeat(*([1, nb_visits, 1])).reshape(nb_patients * nb_visits,
                                                                    litmodel.model.latent_dimension_s)
        rb1_z_s_PI_logvar = b2_z_s_PI_logvar.repeat(*([1, nb_visits, 1])).reshape(nb_patients * nb_visits,
                                                                    litmodel.model.latent_dimension_s)
        # Sample
        rb1_z_s_PI = reparametrize(rb1_z_s_PI_mu, rb1_z_s_PI_logvar)

    # If cannot be batched
    elif nb_visits == 0 or evaluate:
        # Get positions for split per patient
        lengths = [0] + [len(x) for x in indices_2]
        positions = np.add.accumulate(lengths)
        positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]
        # Call permutation invariance network
        b2_z_s_PI_mu = torch.cat([litmodel.model.pi_network(b2_z_s_mu[pos].unsqueeze(0)) for pos in positions])
        b2_z_s_PI_logvar = torch.cat([litmodel.model.pi_network_logvar(b2_z_s_logvar[pos].unsqueeze(0)) for pos in positions])
        # Clip
        b2_z_s_PI_logvar = torch.clamp(b2_z_s_PI_logvar, litmodel.hparams.cliplogvar_min,
                                       litmodel.hparams.cliplogvar_max)
        # Expand
        rb1_z_s_PI_mu = torch.cat([b2_z_s_PI_mu[i].repeat(len(idx_1), 1) for i, idx_1 in enumerate(indices_1)], axis=0)
        rb1_z_s_PI_logvar = torch.cat([b2_z_s_PI_logvar[i].repeat(len(idx_1), 1) for i, idx_1 in enumerate(indices_1)],axis=0)
        # Sample
        rb1_z_s_PI = reparametrize(rb1_z_s_PI_mu, rb1_z_s_PI_logvar)

    return b1_z_psi, rb1_z_s_PI, \
           b1_obs, b1_times, \
           b2_z_s_PI_mu, b2_z_s_PI_logvar, \
           b1_z_psi_mu, b1_z_psi_logvar
           #rb1_z_s_PI_mu, rb1_z_s_PI_logvar, \




def get_latent_perm_invariance_old(litmodel, indices_1, indices_2, dates, obs_list, nb_patients, nb_visits, nb_space_visits):
    """
    Latent permutation invariance & forward encoding
    indices_1: (list of list) longitudinal indices per subject in the batch
    indices_2: (list of list) spatial indices per subject in the batch
    """

    #print(torch.sum(torch.stack([((x == x) * 1.0 - 1).sum() for x in obs_list])))
    #x = torch.cat(obs_list)
    #print(((x == x) * 1.0 - 1).sum())

    #x = bfull_obs
    #print(((x == x) * 1.0 - 1).sum())


    #x = torch.cat([obs[ind_1] for obs, ind_1 in zip(obs_list, indices_1)])
    #print(((x == x) * 1.0 - 1).sum())

    #TODO if subsample & identity : indices2 has to match indices1
    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        indices_2 = indices_1
        nb_space_visits = nb_visits

    # Stack representations
    nested_t_1 = [torch.stack([t[i] for i in ind], 0) for ind, t in zip(indices_1, dates)]
    #nested_obs_1 = [torch.stack([obs[i] for i in ind], 0) for ind, obs in zip(indices_1, obs_list)]
    b1_obs = torch.cat([obs[ind_1] for obs, ind_1 in zip(obs_list, indices_1)])
    b1_times = torch.cat(nested_t_1)
    #b1_obs = torch.cat(nested_obs_1)  # size = (nb_patients * nb_visits, dim)
    #nested_obs_2 = [torch.stack([obs[i] for i in ind], 0) for ind, obs in zip(indices_2, obs_list)]
    b2_obs = torch.cat([obs[ind_2] for obs, ind_2 in zip(obs_list, indices_2)])
    # b2_times = torch.cat(nested_t_2)  # unused
    #b2_obs = torch.cat(nested_obs_2)
    bfull_obs = torch.cat([b1_obs, b2_obs])

    assert not torch.isnan(bfull_obs).any(), "NaN detected bfullobs"

    #print((1-bfull_obs==bfull_obs).sum())
    #print(torch.sum(torch.isnan(bfull_obs)))

    litmodel.last_device = b1_obs.device.index
    assert litmodel.last_device == b1_obs.device.index, "Dates and observations are not on the same device !"

    # ============================================
    # A. FORWARD | Common forward pass on stacked inputs
    mean_psi, logvar_psi, mean_s, logvar_s = litmodel.model.encode(bfull_obs)
    assert not torch.isnan(mean_psi).any(), "NaN detected encoding"
    logvar_psi = torch.clamp(logvar_psi, litmodel.hparams.cliplogvar_min, litmodel.hparams.cliplogvar_max)
    logvar_s = torch.clamp(logvar_s, litmodel.hparams.cliplogvar_min, litmodel.hparams.cliplogvar_max)

    z_psi = reparametrize(mean_psi, logvar_psi)
    z_s = reparametrize(mean_s, logvar_s)

    # TODO Paul : weird here that we make more computations than we should

    b1_z_s, b2_z_s = torch.split(z_s, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)
    b1_z_psi, b2_z_psi = torch.split(z_psi, split_size_or_sections=[b1_obs.shape[0], b2_obs.shape[0]], dim=0)

    # TODO: here, we should allow some flexibility for Permutation invariance (_PI) step

    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        lengths = [0] + [len(x) for x in indices_2]
        positions = np.add.accumulate(lengths)
        positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]
        b2_z_s_PI = torch.cat(
            [litmodel.model.pi_network(b2_z_s[pos]) for pos in positions])
    else:
        # B.1 Case where we subsample with same number of visits per patient
        if nb_visits > 0:
            # Reshape
            b2_z_s_view = b2_z_s.view(*[nb_patients, nb_space_visits, litmodel.model.latent_dimension_s])

            # Call permutation invariance network
            #if litmodel.model.model_name == 'referentialdiffeomorphic_vae':
            b2_z_s_PI = litmodel.model.pi_network(b2_z_s_view)
            #else:
                # TODO: By default, perform mean
            #    b2_z_s_PI = torch.mean(b2_z_s_view, dim=1, keepdim=True)

        # B.2 Case where we no subsampling was performed : potentially different number of visits per patient
        # Cannot be batched
        else:
            lengths = [0]+[len(x) for x in indices_2]
            positions = np.add.accumulate(lengths)
            positions = [range(positions[i], positions[i+1]) for i in range(len(positions)-1)]

            # Call permutation invariance network
            b2_z_s_PI = torch.cat(
                [litmodel.model.pi_network(b2_z_s[pos].unsqueeze(0)) for pos in positions])

            #if litmodel.model.model_name == 'referentialdiffeomorphic_vae':
            #else:
            #    # TODO: By default, perform mean
            #    b2_z_s_PI = torch.cat([torch.mean(b2_z_s[pos].unsqueeze(0), dim=1, keepdim=True) for pos in positions])


    # TODO here depends if subsample or not

    if isinstance(litmodel.model.pi_network, IdentityPermutation):
        rb1_z_s_PI = b2_z_s_PI.reshape(b1_z_psi.shape[0], -1)
    else:
        # Need to repeat over the psi dimension
        # C.1
        if nb_visits > 0:
            rb1_z_s_PI = b2_z_s_PI.repeat(*([1, nb_visits, 1])).reshape(nb_patients * nb_visits,
                                                                        litmodel.model.latent_dimension_s)
        # C.2
        else:
            # here if number of visits is different
            rb1_z_s_PI = torch.cat([b2_z_s_PI[i].repeat(len(idx_1), 1) for i, idx_1 in enumerate(indices_1)], axis=0)


    # TODO, also, where to put the identity ?

    return b1_z_psi, rb1_z_s_PI, \
           b1_obs, b1_times, \
           mean_s, logvar_s, \
           mean_psi, logvar_psi # TODO add here psi_sampled all visits


