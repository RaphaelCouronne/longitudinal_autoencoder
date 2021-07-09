from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch
from src.support.models_helper import reparametrize, compute_kl
from src.support.models_helper import get_modelrelated_hparams, get_attachment_loss, moving_averager, gpu_numpy_detach
from src.support.plotting_helper import plot_patients
import numpy as np
import sys
from copy import deepcopy
from sklearn.feature_selection import mutual_info_regression
from src.support.plotting_helper import plot_trajectory
import itertools
from src.models.longitudinal_models.mlvae import MLVAE
from src.models.longitudinal_models.max_ae import MaxAE
from src.models.longitudinal_models.max_vae import MaxVAE
from src.models.longitudinal_models.bvae import BVAE
from src.models.longitudinal_models.long_vae import LongVAE
from src.models.longitudinal_models.diffeo_vae import DVAE, DRVAE
from src.models.longitudinal_models.bvae_regression import BVAE_Regression
from src.models.longitudinal_models.vae_lssl import VaeLSSL
from functools import reduce
from operator import mul
from src.support.metrics_helper import compute_spearman_metric
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import scipy.stats as stats
import os

from src.support.models_helper import get_latent_perm_invariance, get_indices, compute_soft_spearman


class LongitudinalModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = globals()[self.hparams.model_name](self.hparams.data_info,
                                                        self.hparams.latent_dimension,
                                                        self.hparams.data_statistics,
                                                        **{key: self.hparams[key] for key in
                                                           get_modelrelated_hparams(self.hparams.model_name)}
                                                        )

        # --- Get parameters

        # Generics
        self.type = torch.cuda.FloatTensor if self.hparams.cuda else torch.FloatTensor
        self.random_select = self.hparams.random_select
        self.model.reparametrize = reparametrize
        self.r = np.random.RandomState(self.hparams.random_seed)

        # Losses
        self.att_loss = get_attachment_loss(self.hparams.att_loss)
        self.w = {
            "att": self.hparams.w_att,
            "kl": self.hparams.w_kl,
            "spearman": self.hparams.w_spearman,
            "clr": self.hparams.w_clr,
            "cosine": 1  # self.hparams.w_cosine,
        }

        self.clr = {
            "use": self.hparams.use_clr
        }

        self.softrank = {
            "reg": self.hparams.param_softrank,
            "use": self.hparams.use_softrank
        }

        # GECO training
        if self.hparams.use_GECO:
            self.kappa = deepcopy(self.hparams.kappa)
            self.alpha_smoothing = deepcopy(self.hparams.alpha_smoothing)
            self.moving_avg = None

    ####################################
    ####### IO / Optim #################
    ####################################

    def configure_optimizers(self):
        # if self.model.model_name == "max_ae":
        #    gen_optimizer = Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-3)
        # else:
        gen_optimizer = Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=0)
        gen_sched = {'scheduler': ExponentialLR(gen_optimizer, 0.99),
                     'interval': 'epoch'}  # called after each training step
        return {"optimizer": gen_optimizer, "lr_scheduler": gen_sched}

    def on_after_backward(self):
        """
        Hyper-parameters update at batch_level | GECO updates
        """

        # -------- GECO UPDATES
        if self.hparams.use_GECO:
            if self.trainer.current_epoch and self.global_step % self.hparams.update_every_batch == 0:
                self.w['att'] *= float(np.clip(np.exp(self.moving_avg), 0.99, 1.01))
                self.w['att'] = np.clip(self.w['att'], 1e-6, 1e6)  # safety manual clipping

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, strict=True):
        return self.model.load_state_dict()

    @staticmethod
    def _load_model_state(checkpoint, strict=True, **kwargs):
        args = checkpoint["hyper_parameters"]
        litmodel = LongitudinalModel(args)
        litmodel.model.load_state_dict(checkpoint["state_dict"])
        return litmodel

    ####################################
    ####### Train/Val ##################
    ####################################

    def _step(self, batch, batch_idx, evaluate=False):
        # Compute step
        if self.model.model_type == "agnostic":
            z, x_hat, losses, indices = self._step_longvae(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "diffeo":
            z, x_hat, losses, indices = self._step_diffeo(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "vae":
            z, x_hat, losses, indices = self._step_vae(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "vae_regr":
            z, x_hat, losses, indices = self._step_vae_regr(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "vae_lssl":
            z, x_hat, losses, indices = self._step_vae_lssl(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "max_vae":
            z, x_hat, losses, indices = self._step_maxvae(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "mlvae":
            z, x_hat, losses, indices = self._step_expvae(batch, batch_idx, evaluate=evaluate)
        elif self.model.model_type == "max_ae":
            z, x_hat, losses, indices = self._step_maxae(batch, batch_idx, evaluate=evaluate)
        return z, x_hat, losses, indices

    def training_step(self, batch, batch_idx):
        z, x_hat, losses, indices = self._step(batch, batch_idx)
        b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times = z

        # Losses | modified if GECO update
        if self.hparams.use_GECO:
            constraint = (losses["attachment"] - self.kappa ** 2)
            total_loss = self.w["att"] * constraint \
                         + self.w["kl"] * losses["kl"] \
                         + self.w["spearman"] * losses["spearman"] \
                         + losses["age_regression"] \
                         + self.w["cosine"] * losses["cosine"] \
                         + self.w["clr"] * losses["clr"]
            self.moving_avg = float(gpu_numpy_detach(moving_averager(self.alpha_smoothing, constraint,
                                                                     self.moving_avg, not self.trainer.current_epoch)))
        else:
            total_loss = self.w["att"] * losses["attachment"] \
                         + self.w["kl"] * losses["kl"] \
                         + self.w["spearman"] * losses["spearman"] \
                         + losses["age_regression"] \
                         + self.w["cosine"] * losses["cosine"] \
                         + self.w["clr"] * losses["clr"]

        # Losses
        # self.log("train/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        for key, val in losses.items():
            self.log("train/{}".format(key), val, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/mse", losses["attachment"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/kl", losses["kl"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/spearman", losses["spearman"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train/clr", losses["clr"], prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # Details
        self.log("train_details/w_att", self.w["att"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train_details/w_kl", self.w["kl"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train_details/w_spearman", self.w["spearman"], prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)

        self.logger.experiment.add_histogram("latent_psi/mean", mean_psi, self.global_step)
        self.logger.experiment.add_histogram("latent_psi/logvar", logvar_psi, self.global_step)
        self.logger.experiment.add_histogram("latent_s/mean", mean_s, self.global_step)
        self.logger.experiment.add_histogram("latent_s/logvar", logvar_s, self.global_step)

        return {"loss": total_loss, "z": z}

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        # Compute step
        if self.model.model_type == "agnostic":
            z, x_hat, losses, indices = self._step_longvae(batch, batch_idx)
        elif self.model.model_type == "diffeo":
            z, x_hat, losses, indices = self._step_diffeo(batch, batch_idx)
        elif self.model.model_type == "vae":
            z, x_hat, losses, indices = self._step_vae(batch, batch_idx)
        elif self.model.model_type == "vae_regr":
            z, x_hat, losses, indices = self._step_vae_regr(batch, batch_idx)
        elif self.model.model_type == "vae_lssl":
            z, x_hat, losses, indices = self._step_vae_lssl(batch, batch_idx)
        elif self.model.model_type == "max_vae":
            z, x_hat, losses, indices = self._step_maxvae(batch, batch_idx)
        elif self.model.model_type == "max_ae":
            z, x_hat, losses, indices = self._step_maxae(batch, batch_idx)
        elif self.model.model_type == "mlvae":
            z, x_hat, losses, indices = self._step_expvae(batch, batch_idx)

        # Losses | modified if GECO update
        if self.hparams.use_GECO:
            constraint = (losses["attachment"] - self.kappa ** 2)
            total_loss = self.w["att"] * constraint \
                         + self.w["kl"] * losses["kl"] \
                         + self.w["spearman"] * losses["spearman"] \
                         + losses["age_regression"] \
                         + self.w["cosine"] * losses["cosine"] \
                         + self.w["clr"] * losses["clr"]
        else:
            total_loss = self.w["att"] * losses["attachment"] \
                         + self.w["kl"] * losses["kl"] \
                         + losses["spearman"] * self.w["spearman"] \
                         + losses["age_regression"] \
                         + self.w["cosine"] * losses["cosine"] \
                         + self.w["clr"] * losses["clr"]

        # Logger
        self.log("val/loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/mse", losses["attachment"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/kl", losses["kl"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/spearman", losses["spearman"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/clr", losses["clr"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {"loss": total_loss, "z": z}

    def validation_epoch_end(self, outputs):
        pass

    ####################################
    ####### STEPS ######################
    ####################################

    def _step_expvae(self, batch, batch_idx, evaluate=False):
        pixel_size = self.hparams["data_info"]["total_dim"]
        obs = torch.cat(batch["obs"])
        nb_obs = obs.shape[0]
        mean_psi, logvar_psi, mean_s, logvar_s = self.model.encode(obs)
        mean_pa_list, logvar_pa_list = [], []
        nb_visits = 0
        nb_patients = len(batch["id"])
        self.last_device = obs.device.index
        indices_1, indices_2 = batch["idx_pa"], batch["idx_pa"]

        # %% Compute the zs_mu, zs_logvar per patient (as in MLVAE)
        for idx_pa in batch["idx_pa"]:
            logvar_inv = 1 / torch.exp(logvar_s[idx_pa])
            mean_logvarinv = mean_s[idx_pa] * logvar_inv

            logvar_pa_inv = logvar_inv.sum(axis=0)
            logvar_pa = 1 / logvar_pa_inv
            mean_pa = (mean_logvarinv).sum(axis=0) * logvar_pa

            mean_pa_list.append(mean_pa)
            logvar_pa_list.append(logvar_pa)

        mean_pa = torch.stack(mean_pa_list)
        logvar_pa = torch.stack(logvar_pa_list)

        # Reparametrize
        # zs = reparametrize(mean_pa, logvar_pa)
        zpsi = reparametrize(mean_psi, logvar_psi)

        zs = []

        z = torch.empty(size=(obs.shape[0], self.model.latent_dimension)).type(self.type)
        for i, idx_pa in enumerate(batch["idx_pa"]):
            zs_i = reparametrize(mean_pa[i].repeat((len(idx_pa), 1)), logvar_pa[i].repeat((len(idx_pa), 1)))
            z[idx_pa] = torch.cat([zpsi[idx_pa], zs_i], axis=1)
            zs.append(zs_i)
        zs = torch.cat(zs)

        # %% Losses
        x_hat = self.model.decode(z)

        # Losses
        attachment_loss_ = self.att_loss(x_hat, obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (nb_obs)  # averaged per patient per visit

        kl_loss_s = compute_kl(mean_pa, logvar_pa, type=self.type) / (mean_pa.shape[0] * pixel_size)
        kl_loss_t = compute_kl(mean_psi, logvar_psi, type=self.type) / (nb_obs * pixel_size)
        kl_loss = kl_loss_s + kl_loss_t

        b1_times = torch.cat(batch["t"])
        b1_z_psi = z[:, 0].reshape(-1, 1)
        spearman_loss = compute_soft_spearman(self, indices_1, b1_times, b1_z_psi, nb_visits, nb_patients)

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": spearman_loss,
            "cosine": 0,
            "age_regression": 0,
            "clr": 0,
        }

        rb1_z_s_PI = z[:, 1:]

        # mean_s = z[:, 1:]
        # logvar_s = z[:, 1:]

        if evaluate:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times, mean_psi, mean_s)
        else:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times)

        return z, x_hat, losses, (batch["idx_pa"], batch["idx_pa"])

    def _step_maxae(self, batch, batch_idx, evaluate=False):
        pixel_size = self.hparams["data_info"]["total_dim"]
        b1_obs = torch.cat(batch["obs"])
        x = self.model.encode(b1_obs)
        x = self.model.boost_mlp(x)

        # Do the specific
        z_pa = []
        z_mu_list = []
        z_logvar_list = []
        for idx_pa, t in zip(batch["idx_pa"], batch["t"]):
            x_temp = torch.cat([t.reshape(-1, 1).type(self.type), x[idx_pa]], axis=1)
            z_temp = self.model.rnn(x_temp.unsqueeze(1))[1].reshape(1, -1)
            # z_temp = self.model.boost_mlp(z_temp)
            z_mu, z_logvar = self.model.mlp(z_temp)
            # z_temp = reparametrize(z_mu, z_logvar)
            z_temp = z_mu  # HERE NO VARIATIONAL
            psi_temp = torch.exp(z_temp[:, 0]) * (t.type(self.type) - z_temp[:, 1])
            zs_repeated = z_temp[:, 2:].repeat(len(psi_temp), 1)
            z_pa.append(torch.cat([psi_temp.reshape(-1, 1), zs_repeated], axis=1))
            z_mu_list.append(z_mu)
            # z_logvar_list.append(z_logvar)

        z = torch.cat(z_pa)
        z_mu = torch.cat(z_mu_list)
        z_logvar = torch.ones_like(z_mu)

        # Decode
        b1_obs_pred = self.model.decode(z)

        # Additional constant
        nb_obs = b1_obs_pred.shape[0]

        # Losses
        attachment_loss_ = self.att_loss(b1_obs_pred, b1_obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (nb_obs)  # averaged per patient per visit

        # kl_loss = compute_kl(z_mu, z_logvar, type=self.type) / (z_mu.shape[0] * pixel_size)

        reg = torch.sum(z_mu ** 2) / (z_mu.shape[0] * pixel_size)

        # kl_loss_t = compute_kl(mean_psi, logvar_psi, type=self.type) / (nb_obs * pixel_size)
        # kl_loss = kl_loss_s + kl_loss_t

        losses = {
            "attachment": attachment_loss,
            "kl": reg,
            "spearman": 0,
            "cosine": 0,
            "age_regression": 0,
            "clr": 0
        }

        b1_times = torch.cat(batch["t"])
        b1_z_psi = z[:, 0].reshape(-1, 1)
        rb1_z_s_PI = z[:, 1:]

        mean_s = z_mu[:, 2:]
        logvar_s = z_logvar[:, 2:]

        mean_psi = z_mu[:, 1]
        logvar_psi = z_logvar[:, 1]

        # TODO CHANGE
        if evaluate:
            z = (
            b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times, z[:, 0].reshape(-1, 1), z[:, 1:])
        else:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times)

        return z, b1_obs_pred, losses, (batch["idx_pa"], batch["idx_pa"])

    def _step_maxvae(self, batch, batch_idx, evaluate=False):
        pixel_size = self.hparams["data_info"]["total_dim"]
        b1_obs = torch.cat(batch["obs"])
        x = self.model.encode(b1_obs)
        x = self.model.boost_mlp(x)

        # Do the specific
        z_pa = []
        z_mu_list = []
        z_logvar_list = []
        z_pa_mu = []
        for idx_pa, t in zip(batch["idx_pa"], batch["t"]):
            x_temp = torch.cat([t.reshape(-1, 1).type(self.type), x[idx_pa]], axis=1)
            z_temp = self.model.rnn(x_temp.unsqueeze(1))[1].reshape(1, -1)
            # z_temp = self.model.boost_mlp(z_temp)
            z_mu, z_logvar = self.model.mlp(z_temp)
            z_temp = reparametrize(z_mu, z_logvar)
            # z_temp = z_mu # HERE NO VARIATIONAL
            psi_temp = torch.exp(z_temp[:, 0]) * (t.type(self.type) - z_temp[:, 1])
            zs_repeated = z_temp[:, 2:].repeat(len(psi_temp), 1)
            z_pa.append(torch.cat([psi_temp.reshape(-1, 1), zs_repeated], axis=1))
            z_mu_list.append(z_mu)

            # version no sampling
            psi_temp_mu = torch.exp(z_mu[:, 0]) * (t.type(self.type) - z_mu[:, 1])
            zs_repeated_mu = z_mu[:, 2:].repeat(len(psi_temp), 1)
            z_pa_mu.append(torch.cat([psi_temp_mu.reshape(-1, 1), zs_repeated_mu], axis=1))

            z_logvar_list.append(z_logvar)

        z_pa_mu = torch.cat(z_pa_mu)

        z = torch.cat(z_pa)
        z_mu = torch.cat(z_mu_list)
        z_logvar = torch.cat(z_logvar_list)
        # z_logvar = torch.ones_like(z_mu)

        # Decode
        b1_obs_pred = self.model.decode(z)

        # Additional constant
        nb_obs = b1_obs_pred.shape[0]

        # Losses
        attachment_loss_ = self.att_loss(b1_obs_pred, b1_obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (nb_obs)  # averaged per patient per visit

        kl_loss = compute_kl(z_mu, z_logvar, type=self.type) / (z_mu.shape[0] * pixel_size)

        # reg = torch.sum(z_mu**2)/(z_mu.shape[0]*z_mu.shape[1]*pixel_size)

        # kl_loss_t = compute_kl(mean_psi, logvar_psi, type=self.type) / (nb_obs * pixel_size)
        # kl_loss = kl_loss_s + kl_loss_t

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": 0,
            "cosine": 0,
            "age_regression": 0,
            "clr": 0
        }

        b1_times = torch.cat(batch["t"])
        b1_z_psi = z[:, 0].reshape(-1, 1)
        rb1_z_s_PI = z[:, 1:]

        mean_s = z_mu[:, 2:]
        logvar_s = z_logvar[:, 2:]

        mean_psi = z_mu[:, 1]
        logvar_psi = z_logvar[:, 1]

        # TODO CHANGE
        if evaluate:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times, z_pa_mu[:, 0].reshape(-1, 1),
                 z_pa_mu[:, 1:])
        else:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times)

        return z, b1_obs_pred, losses, (batch["idx_pa"], batch["idx_pa"])

    def _step_longvae(self, batch, batch_idx, evaluate=False):
        """
        :param batch:
        :param batch_idx:
        :param subsample:
        :return: losses, z, x_hat
        """

        # Visit selection parameters

        num_patients = len(batch["id"])
        dimension = self.hparams["data_info"]["total_dim"]
        pixel_size = self.hparams["data_info"]["total_dim"]
        dates = batch['t']
        observations_list = batch['obs']
        nb_patients = len(observations_list)
        nb_visits = self.random_select
        # nb_visits = self.random_select
        batch_max_visits = np.max([len(t) for t in dates])
        nb_space_visits = self.r.randint(1, 1 + batch_max_visits)
        # nb_space_visits = 5

        # Temporal & spatial indices
        indices_1, indices_2 = get_indices(self, batch, nb_space_visits, evaluate=evaluate)

        # Permutation invariance operation #TODO does not work with indices from batch
        out = get_latent_perm_invariance(self, indices_1, indices_2, dates, observations_list, nb_patients, nb_visits,
                                         nb_space_visits, evaluate=evaluate)

        b1_z_psi, rb1_z_s_PI, b1_obs, b1_times, mean_s, logvar_s, mean_psi, logvar_psi, zpsi_mu, zs_mu = out

        z_cat = torch.cat([b1_z_psi, rb1_z_s_PI], axis=1)

        # Decode
        b1_obs_pred = self.model.decode(z_cat)

        # Additional constant
        nb_obs = b1_obs_pred.shape[0]

        # Losses
        attachment_loss_ = self.att_loss(b1_obs_pred, b1_obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (nb_obs)  # averaged per patient per visit

        kl_loss_s = compute_kl(mean_s, logvar_s, type=self.type) / (mean_s.shape[0] * pixel_size)
        kl_loss_t = compute_kl(mean_psi, logvar_psi, type=self.type) / (mean_psi.shape[0] * pixel_size)
        kl_loss = kl_loss_s + kl_loss_t

        spearman_loss = compute_soft_spearman(self, indices_1, b1_times, b1_z_psi, nb_visits, nb_patients,
                                              evaluate=evaluate)

        # %% SimCLR
        if nb_visits == 0 and self.hparams["pi_mode"] == "identity" and self.clr["use"]:
            idx_pairs = [np.random.choice(np.unique(idx_2), 2, replace=False) for idx_2 in indices_2]
            z_space_1 = torch.stack([rb1_z_s_PI[idx_pa][idx_2[0]] for idx_pa, idx_2 in zip(batch["idx_pa"], idx_pairs)])
            z_space_2 = torch.stack([rb1_z_s_PI[idx_pa][idx_2[1]] for idx_pa, idx_2 in zip(batch["idx_pa"], idx_pairs)])
            from src.support.loss_helper import NTXentLoss
            ent = NTXentLoss(z_space_1.device, len(z_space_1), 1, True)
            loss_CLR = ent(z_space_1, z_space_2) / (len(idx_pairs) * pixel_size)
        else:
            loss_CLR = 0

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": spearman_loss,
            "cosine": 0,
            "age_regression": 0,
            "clr": loss_CLR,
        }

        if evaluate:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times, zpsi_mu, zs_mu)
        else:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times)
        return z, b1_obs_pred, losses, (indices_1, indices_2)

    def _step_vae(self, batch, batch_idx, evaluate=False):

        pixel_size = self.hparams["data_info"]["total_dim"]
        dates = batch['t']
        observations_list = batch['obs']
        nb_patients = len(observations_list)

        observations_list = batch['obs']
        obs = torch.cat(observations_list)
        z_mu, z_logvar = self.model.encode(obs)

        z_sampled = reparametrize(z_mu, z_logvar)

        x_hat = self.model.decode(z_sampled)

        # Losses
        attachment_loss_ = self.att_loss(x_hat, obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (obs.shape[0])  # averaged per patient per visit
        kl_loss = compute_kl(z_mu, z_logvar, type=self.type) / (obs.shape[0] * pixel_size)
        spearman_loss = 0

        # Arbitrary choice of s/t
        zs, logs, zt, logt, = z_mu[:, 1:], z_logvar[:, 1:], z_mu[:, 0].unsqueeze(1), z_logvar[:, 0].unsqueeze(1)
        zs_sampled, zt_sampled = z_sampled[:, 1:], z_sampled[:, 0].unsqueeze(1)
        # TODO Raphael : check metrics

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": spearman_loss,
            "age_regression": 0,
            "cosine": 0,
            "clr": 0,
        }

        if evaluate:
            z = (zt_sampled, zs_sampled, zs, logs, zt, logt, zt_sampled, zt, z_mu)
        else:
            z = (zt_sampled, zs_sampled, zs, logs, zt, logt, zt_sampled)

        return z, x_hat, losses, (batch["idx_pa"], batch["idx_pa"])

    def _step_vae_regr(self, batch, batch_idx, evaluate=False):

        pixel_size = self.hparams["data_info"]["total_dim"]
        dates = batch['t']
        observations_list = batch['obs']
        nb_patients = len(observations_list)

        observations_list = batch['obs']
        obs = torch.cat(observations_list)
        z_mu, z_logvar, r_mu, r_logvar = self.model.encode(obs)

        z_sampled = reparametrize(z_mu, z_logvar)
        r_sampled = reparametrize(r_mu, r_logvar)

        prior_z_mu = self.model.u(r_sampled)
        prior_z_logvar = self.model.u_var(r_sampled)

        x_hat = self.model.decode(z_sampled)

        # Losses
        attachment_loss_ = self.att_loss(x_hat, obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (obs.shape[0])  # averaged per patient per visit
        kl_loss = compute_kl(z_mu, z_logvar, type=self.type, prior_mu=prior_z_mu, prior_logvar=prior_z_logvar) / (
                    obs.shape[0] * pixel_size)
        spearman_loss = 0
        age_mse = torch.nn.MSELoss()
        # age_regression_loss = age_mse(r_sampled.reshape(-1), torch.cat(batch["t"]))/pixel_size#/(obs.shape[0] * pixel_size)
        age_regression_loss = torch.sum((r_sampled.reshape(-1) - torch.cat(batch["t"]).type(self.type)) ** 2) / (
                    obs.shape[0] * pixel_size)

        # Arbitrary choice of s/t
        zs, logs, zt, logt = z_mu, z_logvar, r_mu, r_logvar
        zs_sampled, zt_sampled = z_sampled, r_sampled
        # TODO Raphael : check metrics

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": spearman_loss,
            "age_regression": age_regression_loss,
            "cosine": 0,
            "clr": 0,
        }
        if evaluate:
            z = (zt_sampled, zs_sampled, zs, logs, zt, logt, zt_sampled, r_mu, z_mu)
        else:
            z = (zt_sampled, zs_sampled, zs, logs, zt, logt, zt_sampled)

        return z, x_hat, losses, (batch["idx_pa"], batch["idx_pa"])

    def _step_vae_lssl(self, batch, batch_idx, evaluate=False):

        pixel_size = self.hparams["data_info"]["total_dim"]
        dates = batch['t']
        observations_list = batch['obs']
        nb_patients = len(observations_list)

        observations_list = batch['obs']
        obs = torch.cat(observations_list)
        z_mu, z_logvar = self.model.encode(obs)

        z_sampled = reparametrize(z_mu, z_logvar)
        x_hat = self.model.decode(z_sampled)

        # Losses
        attachment_loss_ = self.att_loss(x_hat, obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (obs.shape[0])  # averaged per patient per visit
        kl_loss = compute_kl(z_mu, z_logvar, type=self.type) / (obs.shape[0] * pixel_size)
        spearman_loss = 0

        # TODO cosine loss
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_loss = 0
        weight = 0
        for idx_pa in batch["idx_pa"]:
            combinations = list(itertools.combinations(idx_pa, 2))
            weight += len(combinations)
            for idx_1, idx_2 in combinations:
                z_diff = z_sampled[idx_2] - z_sampled[idx_1].unsqueeze(0)
                cos_loss += cos(z_diff, self.model.u)
        cos_loss /= weight
        cos_loss = 1 - cos_loss

        # Arbitrary choice of s/t
        zs, logs, zt, logt = z_mu, z_logvar, torch.matmul(z_mu, self.model.u.T), torch.matmul(z_logvar, self.model.u.T)
        zs_sampled, zt_sampled = z_sampled, torch.matmul(z_sampled, self.model.u.T)
        # TODO Raphael : check metrics

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": spearman_loss,
            "cosine": cos_loss,
            "age_regression": 0,
            "clr": 0
        }
        if evaluate:
            z = (
            zt_sampled, zs_sampled, zs, logs, zt, logt, torch.cat(batch["t"]), zt, zs)  # TODO: why last = zt_sampled ?
        else:
            z = (zt_sampled, zs_sampled, zs, logs, zt, logt, torch.cat(batch["t"]))

        return z, x_hat, losses, (batch["idx_pa"], batch["idx_pa"])

    def _step_diffeo(self, batch, batch_idx, subsample=False, evaluate=False):
        """
        :param batch:
        :param batch_idx:
        :param subsample:
        :return: losses, z, x_hat
        """

        # Check NAS
        # print([((x == x) * 1.0 - 1).sum() for x in batch["obs"]])

        # Visit selection parameters
        num_patients = len(batch["id"])
        dimension = self.hparams["data_info"]["total_dim"]
        pixel_size = self.hparams["data_info"]["total_dim"]
        dates = batch['t']
        observations_list = batch['obs']
        nb_patients = len(observations_list)
        nb_visits = self.random_select
        batch_max_visits = np.max([len(t) for t in dates])
        nb_space_visits = self.r.randint(1, 1 + batch_max_visits)
        assert nb_patients == len(dates), "Error in compatibility"

        # Temporal & spatial indices
        indices_1, indices_2 = get_indices(self, batch, nb_space_visits, evaluate=evaluate)

        # Permutation invariance operation #TODO does it work withs indices from batch ?
        out = get_latent_perm_invariance(self, indices_1, indices_2, dates, observations_list, nb_patients, nb_visits,
                                         nb_space_visits, evaluate=evaluate)

        b1_z_psi, rb1_z_s_PI, b1_obs, b1_times, mean_s, logvar_s, mean_psi, logvar_psi, zpsi_mu, zs_mu = out

        z = torch.cat([b1_z_psi, rb1_z_s_PI], axis=1)
        b1_obs_pred = self.model.decode(z)

        # TODO because nb_visits can change
        """
        # ============================================
        # LOSSES | attachment_loss_, kl_loss_s, kl_loss_a, ranking_loss (all averaged by pixel)
        # attachment_loss_ is MSE on b1 targets
        attachment_loss_ = self.att_loss(b1_obs_pred, b1_obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (nb_patients * nb_visits)  # averaged per patient per visit

        # TODO Paul pour la KL | careful with the relative weights
        kl_loss_s = compute_kl(mean_s, logvar_s, type=self.type) / (nb_visits * pixel_size)
        kl_loss_t = compute_kl(mean_psi, logvar_psi, type=self.type) / (nb_visits * pixel_size)
        kl_loss = kl_loss_s + kl_loss_t
        """

        # Additional constant
        nb_obs = b1_obs_pred.shape[0]

        # Losses
        attachment_loss_ = self.att_loss(b1_obs_pred, b1_obs)
        attachment_loss_ /= pixel_size if self.hparams.att_loss == 'mse' else 1.  # batch-sum, averaged by pixel
        attachment_loss = attachment_loss_ / (nb_obs)  # averaged per patient per visit

        kl_loss_s = compute_kl(mean_s, logvar_s, type=self.type) / (nb_obs * pixel_size)
        kl_loss_t = compute_kl(mean_psi, logvar_psi, type=self.type) / (nb_obs * pixel_size)
        kl_loss = kl_loss_s + kl_loss_t

        spearman_loss = compute_soft_spearman(self, indices_1, b1_times, b1_z_psi, nb_visits, nb_patients,
                                              evaluate=evaluate)

        # To common variables
        if evaluate:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times, zpsi_mu, zs_mu)
        else:
            z = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times)

        losses = {
            "attachment": attachment_loss,
            "kl": kl_loss,
            "spearman": spearman_loss,
            "age_regression": 0,
            "cosine": 0,
            "clr": 0,
        }

        return z, b1_obs_pred, losses, (indices_1, indices_2)

    def training_epoch_end(self, outputs):

        self.model.eval()

        # %% Now on all observations / z
        if self.hparams["verbose"] and self.current_epoch % self.hparams["verbose"] == 0:

            batch = next(iter(self.val_dataloader()))
            label = list(batch["time_label"].keys())[0]
            true_tstar_list = []
            t_list = []
            for batch in self.val_dataloader():
                for t_star in batch["time_label"][label]:
                    true_tstar_list.append(t_star)
                for t in batch["t"]:
                    t_list.append(t)

            zpsi_visit_list = []
            zs_visit_list = []

            zpsi_pa_list = []
            zs_pa_list = []

            for batch in self.val_dataloader():
                lengths = [0] + [len(x) for x in batch["idx_pa"]]
                positions = np.add.accumulate(lengths)
                positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]

                batch["obs"] = [x.type(self.type) for x in batch["obs"]]
                z, x_hat, _, indices = self._step(batch, 0, evaluate=True)
                zpsi, zs = z[7], z[8]

                zpsi_visit_list.append(zpsi)
                zs_visit_list.append(zs)
                zpsi_pa_list.append(torch.stack([zpsi[pos].mean(axis=0) for pos in positions]))
                zs_pa_list.append(torch.stack([zs[pos].mean(axis=0) for pos in positions]))

            zpsi_visit = torch.cat(zpsi_visit_list).cpu().detach().numpy()
            zs_visit = torch.cat(zs_visit_list).cpu().detach().numpy()
            zpsi_pa = torch.cat(zpsi_pa_list).cpu().detach().numpy()
            zs_pa = torch.cat(zs_pa_list).cpu().detach().numpy()

            b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times, zpsi_mu, zs_mu = z

            # modify the zs for the model that need it
            if self.model.model_name == "vae_lssl":

                u = self.model.u.detach().cpu().numpy()

                zs_visit = zs_visit - np.matmul(zs_visit, u.T) * u / np.linalg.norm(u) ** 2
                zs_pa = zs_pa - np.matmul(zs_pa, u.T) * u / np.linalg.norm(u) ** 2

            elif self.model.model_name == "BVAE_Regression":
                u = self.model.u.weight.data.detach().cpu().numpy()

                zs_visit = zs_visit - np.matmul(zs_visit, u) * u.T / np.linalg.norm(u) ** 2
                zs_pa = zs_pa - np.matmul(zs_pa, u) * u.T / np.linalg.norm(u) ** 2
            elif self.model.model_name == "BVAE":
                # TODO do a PLS here
                pass

            # MI | Compute mutual information
            mi = mutual_info_regression(zs_pa, zpsi_pa).mean()
            self.log("train_disentangle/mi_patients", mi, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # Corr | Compute correlation
            latent_corr = []
            for latent_dim in range(zs_visit.shape[1]):
                latent_corr.append(np.abs(np.corrcoef(zs_pa[:, latent_dim], zpsi_pa.reshape(-1))[0, 1]))
            corr = np.mean(latent_corr)
            self.log("train_disentangle/corr", corr, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # PLS for z psi pa / zs pa
            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs_pa, zpsi_pa)
            Y_pred = pls2.predict(zs_pa)
            pls = np.corrcoef(Y_pred.reshape(-1), zpsi_pa.reshape(-1))[0, 1]
            self.log("train_disentangle/pls_patients", pls, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # PLS For z psi visits / zs Visits
            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs_visit, zpsi_visit)
            Y_pred = pls2.predict(zs_visit)
            pls = np.corrcoef(Y_pred.reshape(-1), zpsi_visit.reshape(-1))[0, 1]
            self.log("train_disentangle/pls_visits", pls, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # PLS with true psi / zs visits
            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs_visit, torch.cat(t_list).detach().cpu().numpy())
            Y_pred = pls2.predict(zs_visit)
            pls = np.corrcoef(Y_pred.reshape(-1), torch.cat(t_list).detach().cpu().numpy().reshape(-1))[0, 1]
            self.log("train_disentangle/pls_visit_truepsi", pls, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # PLS with true psi / zs patients
            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs_pa, np.concatenate([[t.mean()] for t in t_list]))
            Y_pred = pls2.predict(zs_pa)
            pls = np.corrcoef(Y_pred.reshape(-1), np.concatenate([[t.mean()] for t in t_list]))[0, 1]
            self.log("train_disentangle/pls_patients_truepsi", pls, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # MI | Compute mutual information
            mi = mutual_info_regression(zs_visit, zpsi_visit).mean()
            self.log("train_disentangle/mi_visits", mi, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # Corr | Compute correlation
            latent_corr = []
            for latent_dim in range(zs_visit.shape[1]):
                latent_corr.append(np.abs(np.corrcoef(zs_visit[:, latent_dim], zpsi_visit.reshape(-1))[0, 1]))
            corr = np.mean(latent_corr)
            self.log("train_disentangle/corr", corr, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # PCA
            if self.model.latent_dimension > 2:
                pca = PCA(n_components=2)
                pca.fit(zs_visit)
            else:
                pca = PCA(n_components=1)
                pca.fit(zs_visit)

            ####
            # Assess staging quality
            ####

            true_tstar = np.concatenate(true_tstar_list)
            idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
            staging_corr_visits = torch.tensor(
                [float(stats.spearmanr(np.array(true_tstar)[idx_no_na], zpsi_visit[idx_no_na])[0])])
            self.log("val/spearman_t*_visits", staging_corr_visits, prog_bar=True, logger=True, on_step=False,
                     on_epoch=True)

            true_tstar = np.array([np.mean([x]) for x in true_tstar_list])
            idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
            staging_corr_visits = torch.tensor(
                [float(stats.spearmanr(np.array(true_tstar)[idx_no_na], zpsi_pa[idx_no_na])[0])])
            self.log("val/spearman_t*_patients", staging_corr_visits, prog_bar=True, logger=True, on_step=False,
                     on_epoch=True)

            """
            idx_patients = []
            max_idx = 0
            for batch in self.val_dataloader():
                print(max_idx)
                idx_patients += [(np.array(x)+max_idx).tolist() for x in batch["idx_pa"]]
                max_idx = np.max(idx_patients)


            # Intra visits
            pred_tstar = zpsi_visit
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
            self.log("val/spearman_t*_intra", staging_corr_intra, prog_bar=True, logger=True, on_step=False, on_epoch=True)"""

        if self.hparams["verbose"] and self.current_epoch % self.hparams["verbose"] == 0:

            ### For one batch
            # Compute for a batch
            batch = next(iter(self.val_dataloader()))
            batch["obs"] = [x.type(self.type) for x in batch["obs"]]
            z, x_hat, _, indices = self._step(batch, 0, evaluate=True)
            zpsi, zs = z[0], z[1]

            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            # Plot patients
            if self.hparams["data_info"]["dim"] == 1:
                if len(batch["id"]) > 1:
                    obs_list = batch["obs"]
                    x_hat_list = [x_hat[idx_pa] for idx_pa in batch["idx_pa"]]

                    import matplotlib.cm as cm
                    color_palette = cm.get_cmap('tab10', 9)
                    colors = color_palette(range(x_hat.shape[1]))

                    n_patients_to_plot = min(len(obs_list), 4)
                    fig, ax = plt.subplots(1, n_patients_to_plot, figsize=(4 * n_patients_to_plot, 4))
                    for i, ax_x in enumerate(ax):
                        for j in range(x_hat.shape[1]):
                            obs_temp = obs_list[i][:, j].detach().cpu().numpy()
                            x_hat_temp = x_hat_list[i][:, j].detach().cpu().numpy()
                            ax_x.plot(obs_temp, linestyle="--", marker='o', alpha=0.8, c=colors[j])
                            ax_x.plot(x_hat_temp, alpha=0.8, c=colors[j])
                            ax_x.set_ylim(0, 1)
                    self.logger.experiment.add_figure('val/reconst', fig, self.current_epoch)

            elif self.hparams["data_info"]["dim"] == 2:
                if len(batch["id"]) > 1:
                    obs_list = batch["obs"]
                    x_hat_list = [x_hat[idx_pa] for idx_pa in batch["idx_pa"]]
                    try:
                        plot_patients(self, obs_list, x_hat_list, split_name="val")
                    except:
                        pass
            elif self.hparams["data_info"]["dim"] == 3:
                if len(batch["id"]) > 1:

                    slice = int(x_hat.shape[4] / 2)

                    if self.hparams["data_info"]["shape"] == (64, 64, 64):
                        x_hat_list = [x_hat[idx_pa][:, :, :, :, slice] for idx_pa in indices[0]]
                        obs_list = [obs[:, :, :, :, slice] for obs in batch["obs"]]
                    elif self.hparams["data_info"]["shape"] == (91, 109, 91):
                        slice = 41
                        x_hat_list = [x_hat[idx_pa][:, :, slice, :, :] for idx_pa in indices[0]]
                        obs_list = [obs[:, :, slice, :, :] for obs in batch["obs"]]
                    try:
                        plot_patients(self, obs_list, x_hat_list, split_name="val")
                    except:
                        pass

            def build_grad(dim, i, type):
                x = torch.zeros(size=(dim, 1)).type(type)
                x[i] = 1
                return x

            def orthogonal_proj(x, u):
                return x - u * torch.dot(u, x) / (torch.norm(u) ** 2)

            psi_quantile_05 = np.quantile(zpsi_visit, 0.05)
            psi_quantile_95 = np.quantile(zpsi_visit, 0.95)
            psi_quantile_10 = np.quantile(zpsi_visit, 0.1)
            psi_quantile_50 = np.quantile(zpsi_visit, 0.5)
            psi_quantile_90 = np.quantile(zpsi_visit, 0.9)
            if self.model.latent_dimension > 4:
                pca1_quantile_50, pca2_quantile_50 = np.quantile(pca.transform(zs_visit), 0.5, axis=0)
                pca1_quantile_90, pca2_quantile_90 = np.quantile(pca.transform(zs_visit), 0.9, axis=0)
                pls_quantile_50 = np.quantile(pls2.transform(zs_visit), 0.5, axis=0)
                pls_quantile_90 = np.quantile(pls2.transform(zs_visit), 0.9, axis=0)

                deltas = {
                    0: psi_quantile_90 - psi_quantile_50,
                    1: pca1_quantile_90 - pca1_quantile_50,
                    2: pca2_quantile_90 - pca2_quantile_50,
                    3: pls_quantile_90 - pls_quantile_50,
                }
            elif self.model.latent_dimension == 2:
                zs_quantile_50 = np.quantile(zs_visit, 0.5)
                zs_quantile_90 = np.quantile(zs_visit, 0.9)
                deltas = {
                    0: psi_quantile_90 - psi_quantile_50,
                    1: zs_quantile_90 - zs_quantile_50,
                }
            else:
                raise NotImplementedError("No plots for dimensions 3 and 4")

            deltas = {key: torch.tensor([val]).type(self.type) for key, val in deltas.items()}

            if self.model.model_name == "vae_lssl":
                z_average = zs.mean(axis=0)
                z_begin = zs[0].unsqueeze(0)
                # gradients = {0 : self.model.u.clone().detach().reshape(-1)}
                # gradients.update({i: orthogonal_proj(build_grad(self.model.latent_dimension, i, self.type).reshape(-1),
                #                                     self.model.u.reshape(-1))
                #                  for i in range(1, self.model.latent_dimension)})

                if self.model.latent_dimension > 2:
                    gradients = {
                        0: self.model.u.clone().detach().reshape(-1),
                        1: torch.tensor(pca.components_[0]).type(self.type),
                        2: torch.tensor(pca.components_[1]).type(self.type),
                        3: torch.tensor(pls2.coef_.reshape(-1)).type(self.type),
                    }
                else:
                    gradients = {i: build_grad(self.model.latent_dimension, i, self.type).reshape(-1) for
                                 i in range(self.model.latent_dimension)}



            elif self.model.model_name in ["longitudinal_vae", "max_ae", "mlvae",
                                           "max_vae"]:  # referentialdiffeomorphic_vae
                z_average = torch.cat([zpsi.mean(axis=0), zs.mean(axis=0)]).type(self.type)
                z_begin = torch.cat([zpsi[0], zs[0]]).type(self.type).unsqueeze(0)

                if self.model.latent_dimension > 2:
                    gradients = {
                        0: build_grad(self.model.latent_dimension, 0, self.type).reshape(-1),
                        1: torch.tensor(np.concatenate([[0], pca.components_[0]])).type(self.type),
                        2: torch.tensor(np.concatenate([[0], pca.components_[1]])).type(self.type),
                        3: torch.tensor(np.concatenate([[0], pls2.coef_.reshape(-1)])).type(self.type),
                    }
                else:
                    gradients = {i: build_grad(self.model.latent_dimension, i, self.type).reshape(-1) for
                                 i in range(self.model.latent_dimension)}

            elif self.model.model_name in ["referentialdiffeomorphic_vae"]:  # referentialdiffeomorphic_vae
                z_average = torch.cat([zpsi.mean(axis=0), zs.mean(axis=0)]).type(self.type).reshape(1, -1)
                z_begin = torch.cat([zpsi[0], zs[0]]).type(self.type).unsqueeze(0)
                if self.model.latent_dimension > 2:
                    gradients = {
                        0: build_grad(self.model.latent_dimension, 0, self.type).reshape(1, -1),
                        1: torch.tensor(np.concatenate([[0], pca.components_[0]])).type(self.type).reshape(1, -1),
                        2: torch.tensor(np.concatenate([[0], pca.components_[1]])).type(self.type).reshape(1, -1),
                        3: torch.tensor(np.concatenate([[0], pls2.coef_.reshape(-1)])).type(self.type).reshape(1, -1),
                    }
                else:
                    gradients = {i: build_grad(self.model.latent_dimension, i, self.type).reshape(-1) for
                                 i in range(self.model.latent_dimension)}


            elif self.model.model_name in ["BVAE"]:  # TODO for now no regression in laent space

                pca = PCA(n_components=4)
                pca.fit(np.concatenate([zpsi_visit, zs_visit], axis=1))

                z_average = torch.cat([zpsi, zs], axis=1).mean(axis=0)
                z_begin = torch.cat([zpsi, zs], axis=1)[0].unsqueeze(0)

                if self.model.latent_dimension > 2:
                    gradients = {
                        0: torch.tensor(pca.components_[0]).type(self.type),
                        1: torch.tensor(pca.components_[1]).type(self.type),
                        2: torch.tensor(pca.components_[2]).type(self.type),
                        3: torch.tensor(pca.components_[3]).type(self.type),
                    }

                    pca1_q50, pca2_q50, pca3_q50, pca4_q50 = np.quantile(
                        pca.transform(np.concatenate([zpsi_visit, zs_visit],
                                                     axis=1)), 0.5, axis=0)
                    pca1_q90, pca2_q90, pca3_q90, pca4_q90 = np.quantile(
                        pca.transform(np.concatenate([zpsi_visit, zs_visit],
                                                     axis=1)), 0.9, axis=0)

                    deltas = {
                        0: pca1_q90 - pca1_q50,
                        1: pca2_q90 - pca2_q50,
                        2: pca3_q90 - pca3_q50,
                        3: pca4_q90 - pca4_q50,
                    }
                else:
                    gradients = {i: build_grad(self.model.latent_dimension, i, self.type).reshape(-1) for
                                 i in range(self.model.latent_dimension)}





            elif self.model.model_name in ["BVAE_Regression"]:  # TODO for now no regression in laent space
                z_average = zs.mean(axis=0)
                z_begin = zs[0].unsqueeze(0)

                u = self.model.u.weight.data.clone().detach().reshape(-1)
                if self.model.latent_dimension > 2:
                    gradients = {
                        0: u,
                        1: torch.tensor(pca.components_[0]).type(self.type),
                        2: torch.tensor(pca.components_[1]).type(self.type),
                        3: torch.tensor(pls2.coef_.reshape(-1)).type(self.type),
                    }
                else:
                    gradients = {i: build_grad(self.model.latent_dimension, i, self.type).reshape(-1) for
                                 i in range(self.model.latent_dimension)}

                # gradients.update({i: orthogonal_proj(build_grad(self.model.latent_dimension, i, self.type).reshape(-1),
                #                                     u)
                #                  for i in range(1, self.model.latent_dimension)})
            else:
                raise NotImplementedError

            # %% Plot patient latent

            if self.hparams["data_info"]["dim"] == 1:
                n_patients_to_plot = min(len(obs_list), 4)
                fig, ax = plt.subplots(1, 1)

                if self.model.latent_dimension == 2:
                    if self.model.model_name in ["longitudinal_vae"]:
                        z_temp = torch.cat([zpsi, zs], axis=1).detach().cpu().numpy()
                    elif self.model.model_name == "vae_lssl":
                        z_temp = zs.detach().cpu().numpy()

                elif self.model.latent_dimension > 4:
                    if self.model.model_name in ["longitudinal_vae", "max_ae", "max_vae"]:
                        # z_temp = torch.cat([zpsi, zs], axis=1).detach().cpu().numpy()
                        z_temp = np.concatenate([zpsi_visit, pca.transform(zs_visit)], axis=1)
                    elif self.model.model_name == "vae_lssl":
                        # z_temp = torch.cat([zpsi, zs], axis=1).detach().cpu().numpy()
                        z_temp = np.concatenate([zpsi_visit, pca.transform(zs_visit)], axis=1)
                    elif self.model.model_name == "BVAE_Regression":
                        z_temp = torch.cat([zpsi, zs], axis=1).detach().cpu().numpy()
                    elif self.model.model_name == "BVAE":
                        z_temp = zs.detach().cpu().numpy()
                    else:
                        raise NotImplementedError()
                for i, idx_pa in enumerate(batch["idx_pa"]):
                    ax.plot(z_temp[idx_pa, 0], z_temp[idx_pa, 1])
                    if i > n_patients_to_plot:
                        break

                self.logger.experiment.add_figure('val/latent_space', fig, self.current_epoch)

            if self.current_epoch > 5:  # Because of gradients at beginning that do not make sens
                if self.hparams["data_info"]["dim"] == 1:

                    # TODO special case, regression for BVAE
                    if self.model.model_name == "BVAE":
                        z_cat = zs_visit  # np.concatenate([zpsi_visit, zs_visit], axis=1)
                        t_cat = torch.cat(t_list).detach().cpu().numpy()

                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(z_cat, t_cat)

                        u = reg.coef_ / np.linalg.norm(reg.coef_)
                        # pca in orthogonal
                        z_cat_orth = z_cat - np.matmul(z_cat, u).reshape(-1, 1) * u
                        pca = PCA(n_components=1)
                        pca.fit(z_cat_orth)
                        # pca1_bvae = pca.transform(z_cat_orth)

                        gradients = {
                            0: torch.tensor(u).type(self.type),
                            1: torch.tensor(pca.components_[0]).type(self.type),
                            2: torch.tensor(pca.components_[0]).type(self.type),
                            3: torch.tensor(pca.components_[0]).type(self.type),
                        }

                        psi_quantile_05, psi_quantile_95 = np.quantile(np.matmul(z_cat, u), 0.05), np.quantile(
                            np.matmul(z_cat, u), 0.95)
                        pca1_q50 = np.quantile(pca.transform(z_cat_orth), 0.5, axis=0)
                        pca1_q90 = np.quantile(pca.transform(z_cat_orth), 0.5, axis=0)

                        deltas = {
                            0: float(psi_quantile_95 - psi_quantile_05),
                            1: float(pca1_q90 - pca1_q50),
                            2: float(pca1_q90 - pca1_q50),
                            3: float(pca1_q90 - pca1_q50),
                        }

                    # Do time
                    nb_timepoints = 20
                    grid_timepoints = np.linspace(psi_quantile_05, psi_quantile_95, nb_timepoints)
                    traj_z = torch.stack([z_average + t * gradients[0] for t in grid_timepoints])
                    traj_x_hat = self.model.decode(traj_z)

                    num_plots_latent = min(4, self.model.latent_dimension)

                    # Gradient
                    fig, ax = plt.subplots(num_plots_latent, 1)
                    for i in range(num_plots_latent):
                        try:
                            # jacobian = torch.autograd.functional.jvp(self.model.decode, z_average.reshape(1,-1), gradients[i].reshape(1,-1))[1]
                            jacobian = torch.autograd.functional.jvp(self.model.decode, traj_z,
                                                                     gradients[i].repeat(nb_timepoints, 1))[1]
                            jacobian = jacobian.detach().cpu().numpy()
                            norm = mcolors.DivergingNorm(vmin=jacobian.min(), vmax=jacobian.max(), vcenter=0)
                            ax[i].imshow(jacobian.T, cmap="seismic", norm=norm)
                        except:
                            pass
                        ax[i].get_xaxis().set_visible(False)
                        ax[i].get_yaxis().set_visible(False)
                    self.logger.experiment.add_figure('forward_grad', fig, self.current_epoch)

                    # Finite Difference
                    fig, ax = plt.subplots(num_plots_latent, 1, figsize=(8, 22))
                    for i in range(num_plots_latent):
                        for j in range(x_hat.shape[1]):
                            ax[i].plot(grid_timepoints, traj_x_hat.cpu().detach().numpy()[:, j], c=colors[j],
                                       linewidth=5, alpha=0.8)

                    nb_source = 4
                    for i in range(num_plots_latent):
                        for curve_nb in range(nb_source):
                            traj_x_hat_temp = self.model.decode(
                                traj_z + gradients[i] * deltas[i] * ((curve_nb) / nb_source))
                            for j in range(x_hat.shape[1]):
                                ax[i].plot(grid_timepoints, traj_x_hat_temp.cpu().detach().numpy()[:, j], c=colors[j],
                                           alpha=1 - curve_nb / nb_source)

                    for ax_x in ax:
                        ax_x.set_ylim(0, 1)

                    plt.savefig(os.path.join(self.logger.root_dir, "finite_diff_{}.png".format(self.current_epoch)))
                    self.logger.experiment.add_figure('finite_difference', fig, self.current_epoch)


                elif self.hparams["data_info"]["dim"] in [2, 3]:
                    num_plots_latent = min(4, self.model.latent_dimension)

                    # Plot gradient forward
                    fig, ax = plt.subplots(num_plots_latent, 1)
                    for i in range(num_plots_latent):
                        try:
                            jacobian = torch.autograd.functional.jvp(self.model.decode, z_average, gradients[i])[1]
                            if self.hparams["data_info"]["dim"] == 2:
                                jacobian = jacobian[0, 0, :, :].detach().cpu().numpy()
                            else:
                                if self.hparams["data_info"]["shape"] == (64, 64, 64):
                                    slice = int(self.hparams["data_info"]["shape"][2] / 2)
                                    jacobian = jacobian[0, 0, :, :, slice].detach().cpu().numpy()
                                elif self.hparams["data_info"]["shape"] == (91, 109, 91):
                                    slice = 41
                                    jacobian = jacobian[0, 0, 41, :, :].detach().cpu().numpy()
                                else:
                                    raise NotImplementedError
                            norm = mcolors.DivergingNorm(vmin=jacobian.min(), vmax=jacobian.max(), vcenter=0)
                            ax[i].imshow(jacobian, cmap="seismic", norm=norm)
                            ax[i].get_xaxis().set_visible(False)
                            ax[i].get_yaxis().set_visible(False)
                        except:
                            pass
                    plt.savefig(os.path.join(self.logger.root_dir, "forward_grad_{}.png".format(self.current_epoch)))
                    self.logger.experiment.add_figure('forward_grad', fig, self.current_epoch)

                    # Plot finite difference
                    fig, ax = plt.subplots(num_plots_latent, 1)
                    for i in range(num_plots_latent):
                        try:
                            x_hat_1 = self.model.decode(z_begin)
                            x_hat_2 = self.model.decode(z_begin + gradients[i] * deltas[i])
                            x_hat_diff = (x_hat_2 - x_hat_1).detach().cpu().numpy()
                            norm = mcolors.DivergingNorm(vmin=x_hat_diff.min(), vmax=x_hat_diff.max(), vcenter=0)
                            if self.hparams["data_info"]["dim"] == 2:
                                ax[i].imshow(batch["obs"][0][0, 0, :, :].detach().cpu().numpy(), alpha=1,
                                             cmap="gist_yarg")
                                ax[i].imshow(x_hat_diff[0, 0, :, :], cmap="seismic", alpha=0.5, norm=norm)
                            else:

                                if self.hparams["data_info"]["shape"] == (64, 64, 64):
                                    slice = int(self.hparams["data_info"]["shape"][2] / 2)
                                    ax[i].imshow(batch["obs"][0][0, 0, :, :, slice].detach().cpu().numpy(), alpha=1,
                                                 cmap="gist_yarg")
                                    ax[i].imshow(x_hat_diff[0, 0, :, :, slice], cmap="seismic", alpha=0.5, norm=norm)
                                elif self.hparams["data_info"]["shape"] == (91, 109, 91):
                                    slice = 41
                                    ax[i].imshow(batch["obs"][0][0, 0, slice, :, :].detach().cpu().numpy(), alpha=1,
                                                 cmap="gist_yarg")
                                    ax[i].imshow(x_hat_diff[0, 0, slice, :, :], cmap="seismic", alpha=0.5, norm=norm)
                                else:
                                    raise NotImplementedError
                        except:
                            pass

                        ax[i].get_xaxis().set_visible(False)
                        ax[i].get_yaxis().set_visible(False)
                    plt.savefig(os.path.join(self.logger.root_dir, "finite_diff_{}.png".format(self.current_epoch)))
                    self.logger.experiment.add_figure('finite_difference', fig, self.current_epoch)

        if self.hparams["verbose"] and self.current_epoch % self.hparams["verbose"] == 0:

            n_boostrap = 10
            zs_bootstrap_list = []
            # Here pre-permutation
            for i in range(n_boostrap):
                zs_list = []
                for batch in self.val_dataloader():
                    # sub visits set
                    nb_keep_list = [np.random.randint(max(1, len(x) - 1), len(x)) for x in
                                    batch["obs"]]  # [3 for x in batch["obs"]]# [
                    idx_keep = [np.sort(np.random.choice(range(0, len(x)), nb_keep, replace=False)).tolist() for
                                x, nb_keep in zip(batch["obs"], nb_keep_list)]
                    obs = [x[idx].type(self.type) for x, idx in zip(batch["obs"], idx_keep)]
                    times = [x[idx].type(self.type) for x, idx in zip(batch["t"], idx_keep)]
                    acc = np.add.accumulate([0] + nb_keep_list)
                    idx_pa = [range(acc[i], acc[i + 1]) for i in range(len(acc) - 1)]
                    idx_psi_keep = [int(np.random.choice(idx, 1)) for idx in idx_pa]
                    batch["obs"] = obs
                    batch["times"] = times
                    batch["idx_pa"] = idx_pa
                    batch["t"] = [x[idx].type(self.type) for x, idx in zip(batch["t"], idx_keep)]
                    # step
                    z, x_hat, _, indices = self._step(batch, 0, evaluate=True)
                    zpsi, zs = z[7], z[8]
                    zpsi, zs = torch.stack([zpsi[idx] for idx in idx_psi_keep]), torch.stack(
                        [zs[idx] for idx in idx_psi_keep])
                    zs = zs.detach().cpu().numpy()
                    # Remove time part if needed
                    if self.model.model_name == "vae_lssl":
                        u = self.model.u.detach().cpu().numpy()
                        zs = zs - np.matmul(zs, u.T) * u / np.linalg.norm(u) ** 2
                    elif self.model.model_name == "BVAE_Regression":
                        u = self.model.u.weight.data.detach().cpu().numpy()
                        zs = zs - np.matmul(zs, u) * u.T / np.linalg.norm(u) ** 2
                    # get representation
                    zs_list.append(zs)
                zs_bootstrap_list.append(np.concatenate(zs_list))
            zs_bootstrap = np.stack(zs_bootstrap_list)  # bootstrap x patient (1 visit) x dim

            var_intra = zs_bootstrap.var(axis=(0, 2)).mean()
            var_inter = zs_bootstrap.var(axis=(1, 2)).mean()
            ratio = torch.Tensor([var_intra / var_inter])

            self.log("train_disentangle/ratio", ratio, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

        self.model.train()

        """

        :param outputs (dict): outputs['z'] = (b1_z_psi, rb1_z_s_PI, mean_s, logvar_s, mean_psi, logvar_psi, b1_times)
        :return:

        # Check psi encoder
        #obs = batch["obs"][:][0].cuda()
        obs = torch.cat(batch["obs"]).cuda()
        obs_grad = torch.tensor(obs, requires_grad=True)
        z_psi = self.model.encode_time(obs_grad, do_reparametrize=True)
        z_psi.abs().sum().backward()
        fig, ax = plt.subplots(1,1)
        ax.imshow(np.abs(obs_grad.grad.cpu().detach().numpy().sum(axis=(0,1))), cmap="Reds")
        plt.savefig("../plot_fig3_gradpsi.png")
        plt.close()

        # Check source 0 encoder
        obs = batch["obs"][0][0].unsqueeze(0).cuda()
        obs_grad = torch.tensor(obs, requires_grad=True)
        z_space = self.model.encode_space(obs_grad, do_reparametrize=True)
        z_space.sum().backward()
        fig, ax = plt.subplots(1,1)
        ax.imshow(np.abs(obs_grad.grad.cpu().detach().numpy()[0,0]), cmap="Reds")
        plt.savefig("../plot_fig3_gradspace.png")
        plt.close()


        if self.model.model_name in ["longitudinal_vae", "referentialdiffeomorphic_vae", "mlvae"]:
            mean_psi_all, _, mean_zs_all, _ = self.model.encode(torch.cat(batch["obs"]))
            x_hat = self.model.decode(torch.cat([mean_psi_all, mean_zs_all], axis=1))
        elif self.model.model_name in ["vae_lssl"]:
            z_mu, _, = self.model.encode(torch.cat(batch["obs"]))
            mean_psi_all = torch.matmul(z_mu, self.model.u.T)
            mean_zs_all = z_mu - torch.matmul(mean_psi_all, self.model.u)
            x_hat = self.model.decode(z_mu)
        elif self.model.model_name in ["BVAE_Regression"]:
            mean_zs_all, _, mean_psi_all, _ = self.model.encode(torch.cat(batch["obs"]))
            mean_zs_all = mean_zs_all - self.model.u(mean_psi_all)
            x_hat = self.model.decode(mean_zs_all)
        elif self.model.model_name in ["max_vae"]:
            z = self.model.encode_time_space(torch.cat(batch["obs"]), do_reparametrize=True,
                                             idx_pa=batch["idx_pa"], times=batch["t"])
            x_hat = self.model.decode(z)
            mean_psi_all = z[:, 0]
            mean_zs_all = z[:, 1:]
        else:
            raise NotImplementedError
        mean_psi_all = mean_psi_all.detach().cpu().numpy()
        mean_zs_all = mean_zs_all.detach().cpu().numpy()


        # PLS
        from sklearn.cross_decomposition import PLSRegression
        pls2 = PLSRegression(n_components=1)
        pls2.fit(mean_zs_all, mean_psi_all)
        Y_pred = pls2.predict(mean_zs_all)
        pls = np.corrcoef(Y_pred.reshape(-1), mean_psi_all.reshape(-1))[0, 1]
        self.log("val/pls", pls, prog_bar=False, logger=True,
                 on_step=False, on_epoch=True)


        self.model.eval()

        #%% Now on all observations / z
        if self.hparams["verbose"] and self.current_epoch % self.hparams["verbose"] == 0:
            zpsi_visit_list = []
            zs_visit_list = []

            zpsi_pa_list = []
            zs_pa_list = []

            for batch in self.val_dataloader():
                lengths = [0] + [len(x) for x in batch["idx_pa"]]
                positions = np.add.accumulate(lengths)
                positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]

                batch["obs"] = [x.type(self.type) for x in batch["obs"]]
                z, x_hat, _, indices = self._step(batch, 0, evaluate=True)
                zpsi, zs = z[0], z[1]
                zpsi_visit_list.append(zpsi)
                zs_visit_list.append(zs)
                zpsi_pa_list.append(torch.stack([zpsi[pos].mean(axis=0) for pos in positions]))
                zs_pa_list.append(torch.stack([zs[pos].mean(axis=0) for pos in positions]))

            zpsi_visit = torch.cat(zpsi_visit_list).cpu().detach().numpy()
            zs_visit = torch.cat(zs_visit_list).cpu().detach().numpy()
            zpsi_pa = torch.cat(zpsi_pa_list).cpu().detach().numpy()
            zs_pa = torch.cat(zs_pa_list).cpu().detach().numpy()

            ####
            # Assess correlation in latent space
            ####

            # For patients


            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs_pa, zpsi_pa)
            Y_pred = pls2.predict(zs_pa)
            pls = np.corrcoef(Y_pred.reshape(-1), zpsi_pa.reshape(-1))[0, 1]
            self.log("train_disentangle/pls_patients", pls, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # MI | Compute mutual information
            mi = mutual_info_regression(zs_pa, zpsi_pa).mean()
            self.log("train_disentangle/mi_patients", mi, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # Corr | Compute correlation
            latent_corr = []
            for latent_dim in range(zs_visit.shape[1]):
                latent_corr.append(np.abs(np.corrcoef(zs_pa[:, latent_dim], zpsi_pa.reshape(-1))[0, 1]))
            corr = np.mean(latent_corr)
            self.log("train_disentangle/corr", corr, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # For Visits
            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs_visit, zpsi_visit)
            Y_pred = pls2.predict(zs_visit)
            pls = np.corrcoef(Y_pred.reshape(-1), zpsi_visit.reshape(-1))[0, 1]
            self.log("train_disentangle/pls_visits", pls, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # MI | Compute mutual information
            mi = mutual_info_regression(zs_visit, zpsi_visit).mean()
            self.log("train_disentangle/mi_visits", mi, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # Corr | Compute correlation
            latent_corr = []
            for latent_dim in range(zs_visit.shape[1]):
                latent_corr.append(np.abs(np.corrcoef(zs_visit[:, latent_dim], zpsi_visit.reshape(-1))[0, 1]))
            corr = np.mean(latent_corr)
            self.log("train_disentangle/corr", corr, prog_bar=False, logger=True,
                     on_step=False, on_epoch=True)

            # PCA
            pca = PCA(n_components=2)
            pca.fit(zs_visit)


            ####
            # Assess staging quality
            ####

            label = list(batch["time_label"].keys())[0]

            true_tstar_list = []
            for batch in self.val_dataloader():
                for t_star in batch["time_label"][label]:
                    true_tstar_list.append(t_star)

            idx_patients = []
            max_idx = 0
            for batch in self.val_dataloader():
                print(max_idx)
                idx_patients += [(np.array(x)+max_idx).tolist() for x in batch["idx_pa"]]
                max_idx = np.max(idx_patients)


            true_tstar = np.concatenate(true_tstar_list)
            idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
            staging_corr_visits = torch.tensor(
                [float(stats.spearmanr(np.array(true_tstar)[idx_no_na], zpsi_visit[idx_no_na])[0])])
            self.log("val/spearman_t*_visits", staging_corr_visits, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            #true_tstar = np.concatenate([x.mean() for x in true_tstar_list])
            #idx_no_na = np.arange(len(true_tstar))[np.logical_not(np.isnan(true_tstar.reshape(-1)))].astype(int)
            #staging_corr_visits = torch.tensor(
            #    [float(stats.spearmanr(np.array(true_tstar)[idx_no_na], zpsi_pa[idx_no_na])[0])])
            #self.log("val/spearman_t*_patients", staging_corr_visits, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            # Intra visits
            pred_tstar = zpsi_visit
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
            self.log("val/spearman_t*_intra", staging_corr_intra, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            #compute_spearman_metric(true_tstar_list, pred_tstar, idx_patients)

            # Psi : need to recompute all z_psi
            #from src.support.metrics_helper import compute_spearman_metric
            #label = list(batch["time_label"].keys())[0]
            #staging_corr_visits, staging_corr_intra = compute_spearman_metric(batch, label, mean_psi_all)

            #self.log("val/spearman_t*_intra", staging_corr_intra, prog_bar=True, logger=True, on_step=False,
            #         on_epoch=True)

            #from src.support.plotting_helper import plot_psi
            #label = list(batch["time_label"].keys())[0]
            #plot_psi(self, batch, mean_psi_all, label)







        if self.model.model_name in ["longitudinal_vae", "vae_regr", "vae_lssl", "mlvae"]:

            # Do a pass on the data to get the representations #TODO do it in training_step ?
            zs_list = []
            zpsi_list = []
            for batch in self.val_dataloader():
                obs = torch.stack([x[np.random.randint(len(x))] for x in batch["obs"]]).type(self.type)
                zs_temp = self.model.encode_space(obs, do_reparametrize=True)
                zpsi_temp = self.model.encode_time(obs, do_reparametrize=True)
                zs_list.append(zs_temp.detach().cpu().numpy())
                zpsi_list.append(zpsi_temp.detach().cpu().numpy())
            zs = np.concatenate(zs_list)
            zpsi = np.concatenate(zpsi_list)

        # ==================================
        # PLOTS PCA / PLS / AVERAGE

        if self.model.model_name in ["longitudinal_vae", "mlvae"]:

            num_plots_source = 7
            num_plots_psi = 5

            # Average Trajectory
            values_traj_list = []
            zs_mean = zs.mean(axis=0)
            psi_min, psi_max = np.quantile(zpsi, 0.1), np.quantile(zpsi, 0.9)
            for psi in np.linspace(psi_min, psi_max, num_plots_psi):
                z_temp = np.concatenate([np.repeat(psi, 1).reshape(1,-1),
                                         zs_mean.reshape(1,-1)], axis=1)
                values_traj_list.append(torch.tensor(z_temp).type(self.type))

            plot_trajectory(self, values_traj_list, name="Average Trajectory")

            # PLS Regression
            from sklearn.cross_decomposition import PLSRegression
            pls2 = PLSRegression(n_components=1)
            pls2.fit(zs, zpsi)
            zspred = pls2.predict(zs)
            from sklearn.metrics import explained_variance_score
            explained_variance_score(zpsi, zspred) # TODO explained variance to add

            values_traj = np.linspace(np.quantile(pls2.transform(zs), 0.1),
                                      np.quantile(pls2.transform(zs), 0.9), num_plots_source)
            values_traj = np.concatenate([pls2.inverse_transform(x.reshape(-1, 1)) for x in values_traj])
            values_traj = np.concatenate([np.repeat([zpsi.mean()], (len(values_traj))).reshape(-1,1),values_traj],axis=1)
            values_traj_list = [torch.tensor(values_traj).type(self.type)]

            plot_trajectory(self, values_traj_list, name="PLS")

            #%% PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(zs)
            values_traj = np.linspace(np.quantile(pca.transform(zs), 0.1),
                                      np.quantile(pca.transform(zs), 0.9), num_plots_source)
            values_traj = np.concatenate([pca.inverse_transform(x.reshape(-1, 1)) for x in values_traj])
            values_traj_list = []

            psi_min, psi_max = np.quantile(zpsi, 0.1), np.quantile(zpsi, 0.9)
            for psi in np.linspace(psi_min, psi_max, num_plots_psi):
                z_temp = np.concatenate([np.repeat(psi, num_plots_source).reshape(-1,1),
                                         values_traj], axis=1)
                values_traj_list.append(torch.tensor(z_temp).type(self.type))


            plot_trajectory(self, values_traj_list, name="Psi-PCA1")

        # ==================================
        # RATIO COMPUTATION


        if self.hparams["verbose"] and self.current_epoch%self.hparams["verbose"] == 0:
            z_psi_patients = torch.cat([out["z"][0] for out in outputs]).detach().cpu().numpy()
            z_space_patients = torch.cat([out["z"][1] for out in outputs]).detach().cpu().numpy()
            # mean_t_reparam_patients = torch.cat([out["z"]["mean_t_reparam_patients"] for i in range(len(outputs))])

        """
