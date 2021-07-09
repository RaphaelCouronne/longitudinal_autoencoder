import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt


def compute_spearman_metric(true_tstar_list, pred_tstar, idx_patients):
    true_tstar = np.concatenate(true_tstar_list)
    idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
    staging_corr_visits = torch.tensor(
        [float(stats.spearmanr(np.array(true_tstar)[idx_no_na], pred_tstar[idx_no_na])[0])])

    # Intra visits
    true_tstar_list
    idx_no_na = [np.arange(len(x))[np.logical_not(np.isnan(x)).reshape(-1)] for x in true_tstar_list]
    # Per patients without NAs
    true_tstar_nona = [np.array(x)[idx] for x, idx in zip(true_tstar_list, idx_no_na)]
    pred_tstar_nona = [pred_tstar[idx_pa][idx] for
                       idx_pa, idx in zip(idx_patients, idx_no_na)]

    staging_corr_intra = torch.Tensor(
        [np.mean(
            [float(stats.spearmanr(x, y)[0]) for x, y in zip(true_tstar_nona, pred_tstar_nona) if
             len(x) > 2])]).reshape(-1,
                                    1)

    return staging_corr_visits, staging_corr_intra


"""
# Spearman Metri
def compute_spearman_metric(batch, label, psi_hat):

    true_tstar = np.concatenate((batch["time_label"][label]))
    pred_tstar = psi_hat
    idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
    staging_corr_visits = torch.tensor(
        [float(stats.spearmanr(np.array(true_tstar)[idx_no_na], pred_tstar[idx_no_na])[0])])

    # Intra visits
    true_tstar = batch["time_label"][label]
    idx_no_na = [np.arange(len(x))[np.logical_not(np.isnan(x)).reshape(-1)] for x in true_tstar]
    # Per patients without NAs
    true_tstar_nona = [np.array(x)[idx] for x, idx in zip(true_tstar, idx_no_na)]
    pred_tstar_nona = [pred_tstar[idx_pa][idx] for
                       idx_pa, idx in zip(batch["idx_pa"], idx_no_na)]

    staging_corr_intra = torch.Tensor(
        [np.mean(
            [float(stats.spearmanr(x, y)[0]) for x, y in zip(true_tstar_nona, pred_tstar_nona) if
             len(x) > 2])]).reshape(-1,
                                    1)

    return staging_corr_visits, staging_corr_intra




def spearman_metrics(batch, indices, b1_sampled_tstar, label_tstar, random_select):
    true_tstar = np.concatenate([np.array(x)[idx_b1] for x, idx_b1 in zip(batch["time_label"][label_tstar], indices[0])])
    pred_tstar = b1_sampled_tstar.cpu().detach().numpy()
    idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
    staging_corr_visits = torch.tensor([float(stats.spearmanr(np.array(true_tstar)[idx_no_na], pred_tstar[idx_no_na])[0])])

    # Intra visits
    true_tstar = [np.array(x)[idx_b1] for x, idx_b1 in zip(batch["time_label"][label_tstar], indices[0])]
    idx_no_na = [np.arange(len(x))[np.logical_not(np.isnan(x)).reshape(-1)] for x in true_tstar]
    # Per patients without NAs
    true_tstar_nona = [x[idx] for x, idx in zip(true_tstar, idx_no_na)]
    pred_tstar_nona = [b1_sampled_tstar.cpu().detach().numpy()[idx_pa][idx] for
                       idx_pa, idx in zip(indices[0], idx_no_na)]
    # Remove patients where there was multiple sample of same visit
    true_tstar_nona = [x for x, idx in zip(true_tstar_nona, indices[0]) if len(np.unique(idx)) == random_select]
    pred_tstar_nona = [x for x, idx in zip(pred_tstar_nona, indices[0]) if len(np.unique(idx)) == random_select]

    staging_corr_intra = torch.Tensor(
        [np.mean(
            [float(stats.spearmanr(x, y)[0]) for x, y in zip(true_tstar_nona, pred_tstar_nona) if len(x) > 2])]).reshape(-1,
                                                                                                                         1)
    return staging_corr_visits, staging_corr_intra"""