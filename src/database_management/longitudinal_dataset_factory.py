import numpy as np
import os
import pandas as pd
from src.database_management.longitudinal_dataset import LongitudinalDataset
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torchvision import transforms
from torch.utils.data import Dataset
#from src.support.generics import to_bchw, to_bchwd
#from src.database_management import SeriesGaussianNoise, SeriesOneScale



class LongitudinalDatasetFactory:

    @staticmethod
    def build(dataset_input_path, name, version=None, cv_index=0, cv=None, num_visits=int(1e8), random_seed=0, **kwargs):

        #root_lustre = "/Volumes/dtlake01.aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21"
        root_lustre = "/network/lustre/dtlake01/aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21"

        if name == "ADNI_MRI":
            # ADNI MRI
            paths = ['path_3d_128', 'path_3d_64',
                     'path_2d_midaxial_128', 'path_2d_midaxial_64']

            assert version in paths

            data_infos = {
                    'path_3d_128' :
                        {
                            "dim": 3,
                            "shape": (128, 128, 128),
                            "total_dim": 128 * 128 * 128,
                            "scale": 255,
                        },
                  'path_3d_64' :
                        {
                            "dim": 3,
                            "shape": (64, 64, 64),
                            "total_dim": 64 * 64 * 64,
                            "scale": 255,
                        },
                  'path_2d_midaxial_128':
                        {
                            "dim": 2,
                            "shape": (128, 128),
                            "total_dim": 128 * 128,
                            "scale": 255,
                        },
                  'path_2d_midaxial_64' :
                        {
                            "dim": 2,
                            "shape": (64, 64),
                            "total_dim": 64 * 64,
                            "scale": 255,
                        },
            }

            adni_df = "real/ADNI/AD/Images/MRI/mci_adni_infos.tsv"
            df = pd.read_csv(os.path.join(root_lustre, adni_df), sep="\t")
            df = df.sort_values(["participant_id", "age"])
            features = ["adni_ventricles_vol", "adni_brain_vol"]
            df["diagnosis_int"] = pd.factorize(df["diagnosis"])[0]
            for feature in features:
                idx = df[df[features].abs() > 1e10][feature].dropna().index
                df.loc[idx, feature] = np.nan
            df["ratio_ventricules"] = df["adni_ventricles_vol"] / df["adni_brain_vol"]

            # 'sub-ADNI010S0161',

            df_descr = {
                "id": "participant_id",
                "t": "age",
                "path": version,
                "cofactors": ["apoe4", "diagnosis_bl"],
                "time_label": ["adni_ventricles_vol","ratio_ventricules","MMSE", "status", "diagnosis_int"],
                "data_info": data_infos[version],
                }

        elif name == "scalar":

            adni_df = os.path.join(dataset_input_path)
            df = pd.read_csv(os.path.join(root_lustre, adni_df))
            df["TIME"] = (df["TIME"]-df["TIME"].mean())/df["TIME"].std()
            df = df.set_index("ID")

            features = df.columns.values[1:]
            df_descr = {
                "id": "ID",
                "t": "TIME",
                "features": features,
                "cofactors": [],#["APOE4", "PTGENDER", "ABETA_bl", "tau", "xi"],
                "time_label": ["Y0"],
                "data_info":  {
                            "dim": 1,
                            "shape": len(features),
                            "total_dim": len(features),
                            "scale": False,
                        },
            }


        elif name == "ADNI_COG":
            # ADNI MRI
            paths = ['']

            #df_avg = pd.read_csv(os.path.join(data_path, "synthetic", "df_avg.csv"))

            #assert version in paths
            adni_df = "cog/20-10_scalar_cognition_collab/real/df_real_tstar.csv"
            df = pd.read_csv(os.path.join(root_lustre, adni_df))

            adni_df_ip = "cog/20-10_scalar_cognition_collab/real/df_ip_real.csv"
            df_ip = pd.read_csv(os.path.join(root_lustre, adni_df_ip))

            df = df.set_index("ID").join(df_ip.set_index("ID")).reset_index()

            group_df = "cog/20-10_scalar_cognition_collab/_inputs/cofactors.csv"
            df_group = pd.read_csv(os.path.join(root_lustre, group_df))

            indices = np.unique(df["ID"])

            features_cofactors = ["TAU_bl", "ABETA_bl", "Hippocampus_bl", "APOE4", "PTGENDER"]
            df_group = df_group.rename(columns={"RID":"ID"})[["ID"]+features_cofactors].set_index("ID")
            df_group = df_group.groupby("ID").apply(lambda x: x.iloc[0,:])

            df = df.set_index("ID").join(df_group.loc[indices]).reset_index()

            features = ["Y0", "Y1", "Y2", "Y3"]

            indices = df["ID"].unique()
            indices_1, indices_2 = np.split(indices, 2)

            # Sources artificial
            df = df.set_index("ID")
            #df.loc[indices_1, "Y0"] += 0.1
            #df.loc[indices_2, "Y0"] -= 0.1
            df.reset_index()
            df["Y0"] = df["Y0"].clip(0,1)

            # t normalize
            df["TIME"] = (df["TIME"]-df["TIME"].mean())/df["TIME"].std()

            df_descr = {
                "id": "ID",
                "t": "TIME",
                "features": features,
                "cofactors": ["APOE4", "PTGENDER", "ABETA_bl", "tau", "xi"],
                "time_label": ["t_star"],
                "data_info":  {
                            "dim": 1,
                            "shape": 4,
                            "total_dim": 4,
                            "scale": False,
                        },
            }

        elif name == "STARMEN":
            # STARMEN

            starmen_df = os.path.join(dataset_input_path,"starmen","output_random", "df.csv")

            #starmen_df = "synthetic/starmen/output_random/df.csv"
            df = pd.read_csv(os.path.join(root_lustre, starmen_df))
            df["t_star"] = df["alpha"]*(df["t"]-df["tau"])

            df["t"] = (df["t"]-df["t"].mean())/df["t"].std()

            df_descr = {
                "id": "id",
                "t": "t",
                "path": "path",
                "cofactors": ["tau", "alpha"],
                "time_label": ["t_star"],
                "data_info": {
                    "dim": 2,
                    "shape": (64, 64),
                    "total_dim": 64*64,
                    "scale": False,
                },
                "dataset_input_path":dataset_input_path,
            }

        elif name == "PPMI_DAT":

            assert version in ["path_normalized", "path_sliced41_reshaped"]

            # PPMI
            ppmi_df = "real/PPMI/PPMI-DatScan/new_visits.csv"
            df = pd.read_csv(os.path.join(root_lustre, ppmi_df))
            df["ID"] = df["ID"].astype(str)
            ppmi_df_old = "real/PPMI/PPMI-DatScan/visits.csv"
            df_old = pd.read_csv(os.path.join(root_lustre, ppmi_df_old))
            df_descr = {
                "id": "ID",
                "t": "age",
                "path": version,
                "cofactors": ["Cohort", "subject_sex", "Group"],
                "time_label": ["Disease Duration", "DAT", "L", "R", "CAUDATE", "PUTAMEN"],
                "data_info": {
                    "dim": 2,
                    "shape": (64, 64),
                    "total_dim": 64 * 64,
                    "scale": False,
                },
            }

            if version == "path_normalized":
                df_descr["data_info"] = {
                    "dim": 3,
                    "shape": (91, 109, 91),
                    "total_dim": 91*109*91,
                    "scale": False,
                }

            # Remove subject where there was a problem in spect dataframe
            subjects_spect_problems = [3387, 3542, 3706, 3710, 3791, 3953, 4107, 4136, 40543, 40755, 40757, 40760, 40778,
                                       41767, 42164, 42449, 50586]
            df.drop(df[np.isin(df['ID'], subjects_spect_problems)].index, inplace=True)

            # Remove nans
            df_nans = df[pd.isna(df_old).sum(axis=1) > 0]
            df = df.drop(df_nans.index)
            print("Removing visit {} of patient {} because of nans".format(df_nans.index.values, df_nans['ID'].values))

            # Remove bad shapes
            df_badshape_0 = df[df["shape_0"] != 91]
            df_badshape_1 = df[df["shape_1"] != 109]
            df_badshape_2 = df[df["shape_2"] != 91]
            index_bad_shapes = np.unique(df_badshape_0.index.values.tolist() +
                                         df_badshape_1.index.values.tolist() +
                                         df_badshape_2.index.values.tolist())
            print(
                "Removing visit {} of patient {} because of shape".format(index_bad_shapes, df.loc[index_bad_shapes, 'ID']))
            df.drop(index_bad_shapes, inplace=True)

            # Keep only Parkinsonians # TODO later
            df = df[df["subject_cohort"] == "PD"]

            # Keep only visits >3
            idx_longitudinal = (df.groupby("ID").apply(len) >= 3).index[df.groupby("ID").apply(len) >= 3]
            df = df.set_index("ID").loc[idx_longitudinal].reset_index()

        else:
            raise ValueError("Dataset not known")

        # Max visits
        df = df.iloc[:num_visits,:]
        # Remove if only one visit
        id_col = df_descr["id"]
        df = df.groupby(id_col).apply(lambda x: x if len(x) > 1 else pd.DataFrame())
        # TODO BUG ADNI ?
        if "id_participant" in df.columns:
            df = df.drop("id_participant", axis=1)
        if "participant_id" in df.columns:
            df = df.drop("participant_id", axis=1)
        df = df.reset_index()#.drop("level_1", axis=1)

        # Do CV split
        indices = np.unique(df[id_col])
        nb_total_subjects = len(indices)

        if cv is None:
            print('Cross Validation set to None. Returning full dataset twice.')
            patient_id_train = indices
            patient_id_test = indices
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)  # deterministic wrt random_state
            idx_train, idx_test = list(kf.split(np.arange(nb_total_subjects)))[cv_index]
            patient_id_train, patient_id_test = indices[idx_train], \
                                                indices[idx_test]

        print('>> Split of data is : \n',
              '>> train : {:d}\n'.format(len(patient_id_train)),
              '>> test   : {:d}\n'.format(len(patient_id_test))
              )

        df_train = df.set_index(id_col).loc[patient_id_train].reset_index()
        df_test = df.set_index(id_col).loc[patient_id_test].reset_index()

        return LongitudinalDataset(df_train, df_descr),\
               LongitudinalDataset(df_test, df_descr)
