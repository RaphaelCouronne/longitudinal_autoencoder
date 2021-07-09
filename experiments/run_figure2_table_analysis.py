from src.longitudinal_model import LongitudinalModel
import os
import numpy as np
import torch
import argparse
from src.database_management.longitudinal_dataset_factory import LongitudinalDatasetFactory
from torch.utils.data import DataLoader
from src.database_management.utils import custom_collate_fn
from sklearn.cross_decomposition import PLSRegression
import scipy.stats as stats
import pandas as pd


#%% recompute metrics
def compute_metrics(folder):
    torch.manual_seed(0)
    np.random.seed(0)
    checkpoint_path = os.path.join(folder, os.listdir(folder)[-1])
    litmodel = LongitudinalModel.load_from_checkpoint(checkpoint_path).cuda()
    litmodel.model.eval()

    # Load Data
    args = argparse.Namespace(**litmodel.hparams)

    train_dataset, val_dataset = LongitudinalDatasetFactory.build(args.dataset, args.dataset_version,
                                                                  cv=args.cv, cv_index=args.cv_index,
                                                                  num_visits=args.num_visits)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn,
                                drop_last=False, num_workers=args.num_workers)

    zpsi_visit_list = []
    zs_visit_list = []

    zpsi_pa_list = []
    zs_pa_list = []

    mse_list = []

    for batch in val_dataloader:
        pixel_size = litmodel.hparams["data_info"]["total_dim"]
        lengths = [0] + [len(x) for x in batch["idx_pa"]]
        positions = np.add.accumulate(lengths)
        positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]

        batch["obs"] = [x.type(litmodel.type) for x in batch["obs"]]
        z, x_hat, _, indices = litmodel._step(batch, 0, evaluate=True)
        zpsi, zs = z[7], z[8]
        zpsi_visit_list.append(zpsi)
        zs_visit_list.append(zs)
        zpsi_pa_list.append(torch.stack([zpsi[pos].mean(axis=0) for pos in positions]))
        zs_pa_list.append(torch.stack([zs[pos].mean(axis=0) for pos in positions]))

        mse_list.append(litmodel.att_loss(x_hat.type(litmodel.type), torch.cat(batch["obs"]).type(litmodel.type)) / len(x_hat) / pixel_size)

    zpsi_visit = torch.cat(zpsi_visit_list).cpu().detach().numpy()
    zs_visit = torch.cat(zs_visit_list).cpu().detach().numpy()
    zpsi_pa = torch.cat(zpsi_pa_list).cpu().detach().numpy()
    zs_pa = torch.cat(zs_pa_list).cpu().detach().numpy()

    mse = float(torch.mean(torch.stack(mse_list)).detach().cpu().numpy())

    spearman, pls_psi, pls_zpsi = {}, {}, None

    # modify the zs for the model that need it
    if litmodel.model.model_name == "vae_lssl":

        u = litmodel.model.u.detach().cpu().numpy()

        zs_visit = zs_visit - np.matmul(zs_visit, u.T) * u / np.linalg.norm(u) ** 2
        zs_pa = zs_pa - np.matmul(zs_pa, u.T) * u / np.linalg.norm(u) ** 2

    elif litmodel.model.model_name == "BVAE_Regression":
        u = litmodel.model.u.weight.data.detach().cpu().numpy()

        zs_visit = zs_visit - np.matmul(zs_visit, u) * u.T / np.linalg.norm(u) ** 2
        zs_pa = zs_pa - np.matmul(zs_pa, u) * u.T / np.linalg.norm(u) ** 2
    elif litmodel.model.model_name == "BVAE":
        pass



    labels = list(batch["time_label"].keys())

    labels_value = {}

    for label in labels:

        true_tstar_list = []
        t_list = []
        for batch in val_dataloader:
            for t_star in batch["time_label"][label]:
                true_tstar_list.append(t_star)
            for t in batch["t"]:
                t_list.append(t)
        labels_value[label] = np.concatenate(true_tstar_list)


    for key,val in labels_value.items():
        idx_no_na = np.array(range(0, len(val)))[np.logical_not(np.isnan(val))]
        print(key, " ", stats.spearmanr(val[idx_no_na], zpsi_visit.reshape(-1)[idx_no_na]))
        spearman_res = stats.spearmanr(val[idx_no_na], zpsi_visit.reshape(-1)[idx_no_na])[0]
        spearman["spear/"+key] = max(spearman_res, -spearman_res)

    for key,val in labels_value.items():
        idx_no_na = np.array(range(0, len(val)))[np.logical_not(np.isnan(val))]
        pls2 = PLSRegression(n_components=1)
        pls2.fit(zs_visit[idx_no_na], val[idx_no_na])
        Y_pred = pls2.predict(zs_visit[idx_no_na])
        print(key, " ", stats.spearmanr(val[idx_no_na], Y_pred))
        pls_psi["pls_psi/"+key] = stats.spearmanr(val[idx_no_na], Y_pred)[0]


    for key,val in labels_value.items():
        pls2 = PLSRegression(n_components=1)
        pls2.fit(zs_visit, zpsi_visit)
        Y_pred = pls2.predict(zs_visit)
        print("zpsi", " ", stats.spearmanr(val, Y_pred))
        pls_zpsi = stats.spearmanr(val, Y_pred)[0]

    # zs / zpsi
    pls2 = PLSRegression(n_components=1)
    pls2.fit(zs_visit, zpsi_visit)
    Y_pred = pls2.predict(zs_visit)
    print("zs/zpsi", " ", stats.spearmanr(zpsi_visit, Y_pred))
    pls_zszpsi = stats.spearmanr(zpsi_visit, Y_pred)[0]

    result = {"mse": mse,
              "pls_zpsi": pls_zpsi,
              "pls_zs/zpsi" : pls_zszpsi}
    result.update(pls_psi)
    result.update(spearman)
    return result



#%%
dataset_folder = "results/FIG2/STARMEN/"
print(dataset_folder)
cv_iters = 5
models = ["BVAE_kl1.0", "MLVAE","BVAE_Regression","MaxAE", "VaeLSSL","LongVAE_r3_pi-max_1SR1", "DVAE_r3_pi-max_1SR1",  "LongVAE_r3_pi-max_1SR0", "LongVAE_r0_pi-identity_1SR1"]
df_list = []
idx=0
for model in models:
    for cv_iter in range(cv_iters):
        folder = os.path.join(dataset_folder, model, "cv_{}".format(cv_iter), "version_0", "checkpoints")
        metrics_cv = compute_metrics(folder)
        df_cv = pd.DataFrame(metrics_cv, index=[idx])
        df_cv["cv_iter"] = cv_iter
        df_cv["model"] = model
        df_list.append(df_cv)
        idx += 1

#%% Metrics
import math
def signif(x, digit):
    if x == 0:
        return 0
    return round(x, digit - int(math.floor(math.log10(abs(x)))) - 1)
df = pd.concat(df_list)
metrics_mu_list = []
metrics_std_list = []

for model in models:
    metrics_mu_list.append(df[df["model"] == model].groupby("cv_iter").apply(lambda x: x.iloc[-1]).mean())
    metrics_std_list.append(df[df["model"] == model].groupby("cv_iter").apply(lambda x: x.iloc[-1]).std())

df_mu = pd.DataFrame(pd.concat(metrics_mu_list,axis=1))
df_std = pd.DataFrame(pd.concat(metrics_std_list,axis=1))
df_mu.columns = models
df_std.columns = models
df_std.to_csv(os.path.join(dataset_folder, "df_mu.csv"))
df_mu.to_csv(os.path.join(dataset_folder, "df_std.csv"))

df_mu.loc["mse"]=df_mu.loc["mse"]*1000
df_std.loc["mse"]=df_std.loc["mse"]*1000

df_mu = df_mu.applymap(lambda x: signif(x,3))
df_std = df_std.applymap(lambda x: signif(x,3))
df_str = df_mu.round(6).astype(str) + "$\scriptstyle\,\pm\," + df_std.round(6).astype(str) +"$"
df_str.to_csv(os.path.join(dataset_folder, "df_str.csv"))