import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

@torch.no_grad()
def plot_patients(litmodel, obs_list, x_hat_list, split_name=""):

    n_patients_to_plot = 4

    # Grid for each patient
    grid_patients = []
    width_plot = []
    for i in range(n_patients_to_plot):
        to_save_patient = torch.cat([obs_list[i], x_hat_list[i]]).detach().cpu()
        grid_patient = make_grid(to_save_patient, padding=10, normalize=True, range=(0, 1), nrow=len(obs_list[i]),
                                 scale_each=False)
        grid_patients.append(grid_patient)
        width_plot.append(grid_patient.shape[2])

    # Padd others
    max_width_plot = max(width_plot)
    for i in range(n_patients_to_plot):
        if width_plot[i]<max_width_plot:
            grid_patients[i] = torch.cat([grid_patients[i],
                                         torch.zeros(size=
                                                     (grid_patients[i].shape[0],
                                                       grid_patients[i].shape[1],
                                                       max_width_plot-grid_patients[i].shape[2]))],
                                         axis=2)

    grid = torch.cat(grid_patients, axis=1)
    npimg = grid.numpy()
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    if litmodel.logger is not None:
        litmodel.logger.experiment.add_figure('{} reconst'.format(split_name), fig,
                                              litmodel.current_epoch)

    return 0


def plot_psi(litmodel, batch, psi_hat, label):
    t_star = np.concatenate((batch["time_label"][label]))
    fig, ax = plt.subplots(1, 1)
    for i, idx_pa in enumerate(batch["idx_pa"]):
        psi_true_pa = np.array(t_star[idx_pa])
        psi_hat_pa = psi_hat[idx_pa]
        ax.plot(psi_hat_pa, psi_true_pa)
        litmodel.logger.experiment.add_figure('psi', fig, litmodel.current_epoch)

def plot_trajectory(litmodel, z_list, name="unnamed", slice_plot=2, slice_num=None, cmap=None):

    #z = torch.cat(z_list)
    #x_hat = litmodel.model.decode(z)
    #n_psi = len(z_list)
    #n_s = len(z_list[0])

    dim = litmodel.hparams["data_info"]["dim"]
    x_hat_list = [litmodel.model.decode(z) for z in z_list]

    if cmap is None:
        cmap = "gray"

    if dim == 3:
        if slice_num is None:
            slice_num = int(litmodel.hparams["data_info"]["shape"][slice_plot]/2)

        if slice_plot == 0:
            x_hat_list = [x_hat[:,:,slice_num,:,:] for x_hat in x_hat_list]
        elif slice_plot == 1:
            x_hat_list = [x_hat[:,:,:,slice_num,:] for x_hat in x_hat_list]
        elif slice_plot == 2:
            x_hat_list = [x_hat[:,:,:,:,slice_num] for x_hat in x_hat_list]

    npimg_list = []
    for x_hat in x_hat_list:
        grid = make_grid(x_hat, padding=10, normalize=True, range=(0, 1), nrow=1, scale_each=False)
        npimg = grid.detach().cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg_list.append(npimg)
    npimg = np.concatenate(npimg_list, axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    if cmap is not None:
        ax.imshow(npimg, interpolation='nearest', cmap=cmap)
    else:
        ax.imshow(npimg.mean(axis=2), cmap=cmap)

    if litmodel.logger is not None:
        litmodel.logger.experiment.add_figure(name, fig, litmodel.current_epoch)

    return npimg


