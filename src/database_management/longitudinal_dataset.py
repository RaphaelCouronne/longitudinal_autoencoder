import os
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

class LongitudinalDataset(Dataset):
    def __init__(self, df, df_descr):
        # Not both
        assert ("path" in df_descr.keys()) != ("features" in df_descr.keys())

        df_renamed = df.rename(columns={df_descr[key]: key for key in ["id", "t", "path"] if key in list(df_descr.keys())})
        df_renamed = df_renamed.sort_values(["id", "t"])  # TODO order with visits here ?
        self.indices = df_renamed["id"].unique()
        self.df = df_renamed.set_index("id")
        self.df_descr = df_descr
        self.load_from_paths = "path" in self.df_descr.keys()

        self.data_info = df_descr["data_info"]
        # scale between 0 and 255
        self.scale = df_descr["data_info"]["scale"]
        self.epsilon = 1e-6  # TODO ask Paul

    def load_obs(self, id):
        # When loading images
        if self.load_from_paths:
            paths = self.df.loc[id, "path"].tolist()
            paths = [x.replace("/network/lustre/dtlake01/aramis/users/paul.vernhet/Data/synthetic_longitudinal/",
                               self.df_descr["dataset_input_path"],
                               #"/network/lustre/dtlake01/aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21/synthetic/"
                               #"/Volumes/dtlake01.aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21/synthetic/",
                               ) for x in paths]
            #paths = [x.replace("/network/lustre/dtlake01/aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21/real/PPMI/PPMI-DatScan",
            #                   "/Volumes/dtlake01.aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21/real/PPMI/PPMI-DatScan",
            #                   ) for x in paths]
            values = torch.stack([torch.FloatTensor(np.load(x)) for x in paths]).unsqueeze(1)
        # Load directly from dataframe
        else:
            values = torch.FloatTensor(self.df.loc[id, self.df_descr["features"]].values)

        if self.scale:
            values = values / self.scale
        return values

    def __getitem__(self, idx):
        id = self.indices[idx]
        sample = {
            'id': id,
            't': torch.FloatTensor(self.df.loc[id, "t"].tolist()),
            'obs': self.load_obs(id),
            'cofactors': {cof: self.df.loc[id, cof].tolist()[0] for cof in self.df_descr["cofactors"]},
            "time_label": {cof: self.df.loc[id, cof].tolist() for cof in self.df_descr["time_label"]},
            }
        return sample

    def __len__(self):
        return len(self.indices)

    def compute_statistics(self):
        """
        Computes static statistics in an online fashion (using Welford’s method)
        """
        print('>> Computing online static statistics for dataset ...')
        count = 0
        for elts in tqdm(range(self.__len__())):
            sample = self.__getitem__(elts)
            images = sample['obs']
            for elt in images:
                elt = elt.detach().clone()
                if count == 0:
                    current_mean = elt
                    current_var = torch.zeros_like(elt)
                else:
                    old_mean = current_mean.detach().clone()
                    current_mean = old_mean + 1. / (1. + count) * (elt - old_mean)
                    current_var = float(count - 1) / float(count) * current_var + \
                                  1. / (1. + count) * (elt - current_mean) \
                                  * (elt - old_mean)
                count += 1

        mean = current_mean.detach().clone().float()
        std = torch.clamp(torch.sqrt(current_var), self.epsilon).detach().clone().float()
        return mean, std

