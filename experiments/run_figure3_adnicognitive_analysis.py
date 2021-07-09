from src.longitudinal_model import LongitudinalModel
import os
import torch
import argparse
from src.database_management.longitudinal_dataset_factory import LongitudinalDatasetFactory
from torch.utils.data import DataLoader
from src.database_management.utils import custom_collate_fn
import scipy.stats as stats
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression

def remove_last(x):
    x.iloc[-1, -1] = np.nan
    return x

def signif(x, digit):
    if x == 0:
        return 0
    return round(x, digit - int(math.floor(math.log10(abs(x)))) - 1)


def compute_metrics_1D(folder):
    torch.manual_seed(0)
    checkpoint_path = os.path.join(folder, os.listdir(folder)[-1])
    litmodel = LongitudinalModel.load_from_checkpoint(checkpoint_path).cuda()
    litmodel.model.eval()

    # Load Data
    args = argparse.Namespace(**litmodel.hparams)
    args.w_cosine = 1

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

    df_dataset = val_dataloader.dataset.df

    last_visits = []

    for batch in val_dataloader:
        pixel_size = litmodel.hparams["data_info"]["total_dim"]
        lengths = [0] + [len(x) for x in batch["idx_pa"]]
        import numpy as np
        positions = np.add.accumulate(lengths)
        positions = [range(positions[i], positions[i + 1]) for i in range(len(positions) - 1)]

        last_visits+=[x[-1].detach().cpu().numpy() for x in batch["obs"]]

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

    df_dataset["z_psi"] = zpsi_visit
    for i in range(zs_visit.shape[1]):
        df_dataset["z_s{}".format(i)] = zs_visit[:,i]


    last_visits = np.stack(last_visits)
    # Here depends on the model

    mse = float(torch.mean(torch.stack(mse_list)).detach().cpu().numpy())
    a = df_dataset.groupby("id").apply(lambda x: remove_last(x))

    z_list = []

    for i, (id, df_pa) in enumerate(df_dataset.groupby("id")):

        df_pa_train = df_pa.iloc[:-1,:]
        last_t = df_pa.iloc[-1,:]["t"]

        # MODEL #TODO depends on model here
        if litmodel.model.model_name == "longitudinal_vae":
            reg = LinearRegression().fit(df_pa_train["t"].values.reshape(-1, 1),
                                         df_pa_train["z_psi"])
            zpsi_pred = reg.predict(last_t.reshape(-1, 1))
            zs_pred = zs_pa[i]
            z = np.concatenate([zpsi_pred, zs_pred]).reshape(1,-1)

        elif litmodel.model.model_name == "vae_lssl":
            zs_visit_train = df_pa_train[["z_s{}".format(i) for i in range(zs_visit.shape[1])]]
            reg = LinearRegression().fit(df_pa_train["t"].values.reshape(-1, 1),
                                         zs_visit_train)
            z = reg.predict(last_t.reshape(-1, 1)).reshape(1,-1)
        else:
            z = 0
        z_list.append(torch.tensor(z).type(litmodel.type))

        # get traj extrapolated : how to conctenate the 2 ??? it depends on the model

    if litmodel.model.model_name in ["longitudinal_vae", "vae_lssl"]:
        # Decode
        x_hat = litmodel.model.decode(torch.cat(z_list)).cpu().detach().numpy()

        # difference
        dim = x_hat.shape[0]*x_hat.shape[1]
        error = np.sum((x_hat-last_visits)**2, axis=1)/(x_hat.shape[1])
        print("Error in prediction : {0:.4f}+-{1:.4f}".format(error.mean(), error.std()))
    else:
        error = np.array([0])

    df_dataset["psi_proxymean"] = df_dataset[["Y0", "Y1", "Y2", "Y3"]].mean(axis=1)
    spearman_inter = df_dataset[["z_psi", "psi_proxymean"]].corr(method="spearman").values[0,1].round(4)
    spearman_intra = df_dataset.groupby("id").apply(lambda x: stats.spearmanr(x["z_psi"], x["t"])[0]).mean()

    return {
        "mse" : mse,
        "error_extra": error.mean(),
        "spearman_inter": spearman_inter,
        "spearman_intra":spearman_intra,
    }


dataset_folder = "results/FIG3/ADNI_COG"
cv_iters = 5
models = ["LongVAE_r3_pi-max_1SR1", "VaeLSSL"]
df_list = []
i=0
for model in models:
    for cv_iter in range(cv_iters):
        for version in [0]:
            folder = os.path.join(dataset_folder, model, "cv_{}".format(cv_iter), "version_{}".format(version), "checkpoints")
            dict_cv = compute_metrics_1D(folder)
            df_cv = pd.DataFrame(dict_cv, index=[i])
            df_cv["cv_iter"] = cv_iter
            df_cv["version"] = version
            df_cv["model"] = model
            df_list.append(df_cv)
            i+=1

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

df_mu = df_mu.applymap(lambda x: signif(x,3))
df_std = df_std.applymap(lambda x: signif(x,3))
df_str = df_mu.round(6).astype(str) + "$\scriptstyle\,\pm\," + df_std.round(6).astype(str) +"$"
df_str.to_csv(os.path.join(dataset_folder, "df_str.csv"))

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
df.plot.scatter(x="spearman_inter", y="mse", cmap="jet")
for i, df_line in df.groupby("version"):
    ax.text(s=df_line["version"].values[0],
            x=df_line["spearman_inter"].values[0],
            y =df_line["spearman_intra"].values[0], fontsize=20)
ax.text(0.94, 0.6, "yo", fontsize=20)
plt.savefig("../../plot_models_adnicog.png")
plt.close()