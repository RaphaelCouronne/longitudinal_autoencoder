import argparse
import torch
torch.set_deterministic(True)
#torch.backends.cudnn.benchmark = False
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
# from src.database_management.synthetic_ellipse import dSpriteLongitudinal
from src.database_management.utils import custom_collate_fn
import os
import numpy as np
from src.longitudinal_model import LongitudinalModel
from pytorch_lightning.loggers import TensorBoardLogger
from src.database_management.longitudinal_dataset_factory import LongitudinalDatasetFactory



def run_main(args):
    # ==================================================================================================================
    # GPU SETUP | SEEDS
    # ==================================================================================================================

    if args.dataset_input_path == "null":
        raise ValueError("Please state input dataset folder")

    # CPU/GPU settings || random seeds
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.cuda:
        print('>> GPU available.')
        DEVICE = torch.device('cuda')
        torch.cuda.set_device(args.num_gpu)
        torch.cuda.manual_seed(args.random_seed)
    else:
        DEVICE = torch.device('cpu')
        print('>> CUDA is not available. Overridding with device = "cpu".')
        print('>> OMP_NUM_THREADS will be set to ' + str(args.num_threads))
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
        torch.set_num_threads(args.num_threads)

    # Paths & folders
    model_name = args.model_name
    if model_name in ["DVAE", "LongVAE"]:
        model_name = "{}_r{}_pi-{}_1SR{}".format(model_name, args.random_select, args.pi_mode, int(args.use_softrank))

    logger_path = os.path.join(args.folder, args.dataset, model_name)
    logger = TensorBoardLogger(logger_path, name="cv_{}".format(args.cv_index))

    # Data
    train_dataset, val_dataset = LongitudinalDatasetFactory.build(args.dataset_input_path, args.dataset, args.dataset_version,
                                                                  cv=args.cv, cv_index=args.cv_index,
                                                                  num_visits=args.num_visits)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn,
                                  drop_last=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn,
                                drop_last=False, num_workers=args.num_workers)

    # litmodel
    args.df_descr = train_dataset.df_descr
    data_info = train_dataset.data_info
    args.data_info = data_info
    args.data_statistics = train_dataset.compute_statistics()
    litmodel = LongitudinalModel(args)

    # TODO: Pre-training for diffeos

    # Trainer
    trainer = Trainer.from_argparse_args(args, default_root_dir=args.folder,
                                         gpus=([args.num_gpu] if args.cuda else None),
                                         logger=logger)

    print("========================")
    print("=== Launching {} {} {}".format(args.model_name, args.pi_mode, args.random_select))
    print("========================")

    # Launch training
    trainer.fit(litmodel, train_dataloader, val_dataloaders=val_dataloader)





if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser(description='Longitudinal Unsupervised | No-GECO | LIGHTNING VERSION.')
    # General action parameters
    # parser.add_argument('--data_dir', type=str, default='Data', help='Data directory root.')
    parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
    parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use if cuda not available')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers.')
    # Dataset parameters
    parser.add_argument('--dataset_input_path', type=str, default="null", help='Folder where the dataset is stored')
    parser.add_argument('--folder', type=str, default="Experiment", help='Name of folder')
    parser.add_argument('--dataset', type=str, default='mci_adni', help='Dataset choice.')
    parser.add_argument('--dataset_version', type=str, default=None, help='Dataset version.')
    parser.add_argument('--fold', type=int, default=int(0), help='which fold in 5 fold')
    parser.add_argument('--cv', type=int, default=5, help='cross validation fold number.')
    parser.add_argument('--cv_index', type=int, default=0, help='Index of chosen split, for reproducibility.')
    parser.add_argument('--num_visits', type=int, default=int(1e8), help='Maximum number of visits in dataset')
    # Model
    parser.add_argument('--model_name', type=str, default="BVAE", help='Which model to use')
    parser.add_argument('--latent_dimension', type=int, default=2, help='Trajectory Latent dimension.')
    # Architecture
    parser.add_argument('--decoder_last_activation', type=str, default='identity', help='decoder last_function name.')
    parser.add_argument('--one_encoder',action='store_true', help='Whether to use only 1 encoder')
    parser.add_argument('--nn_size', type=str, default='normal', help='Wether to use bigger networks')
    # Subsampling
    parser.add_argument('--random_select', type=int, default=3, help='Random selection of individuals')
    # Loss
    parser.add_argument('--att_loss', type=str, default='mse', help='Attachment loss name.')
    parser.add_argument('--w_att', type=float, default=1.0, help='Weight for attachment loss')
    parser.add_argument('--w_kl', type=float, default=1.0, help='Weight for KL loss')
    parser.add_argument('--w_spearman', type=float, default=0.1, help='Weight for Spearman soft ranking loss')
    parser.add_argument('--w_clr', type=float, default=1.0, help='Weight for Contrastive Loss')
    parser.add_argument('--use_softrank', action='store_true', help='Whether to use soft-rank penalization.')
    parser.add_argument('--use_clr', action='store_true', help='Whether to use clr loss.')
    parser.add_argument('--param_softrank', type=float, default=.25, help='Parameter for regularity for softrank')
    parser.add_argument("--lambda_square", type=float, default=1., help="Gaussian variance prior: space.")
    parser.add_argument("--mu_square", type=float, default=1., help="Gaussian variance prior: time.")
    parser.add_argument('--cliplogvar_min', type=float, default=float(-2 * 2 * np.log(10) + 2*np.log(2)), help='10**min clip variance.')
    parser.add_argument('--cliplogvar_max', type=float, default=float(3 * 2 * np.log(10)), help='10**max clip variance.')
    parser.add_argument("--tol", type=float, default=float(1e-10), help="Numerical tolerance.")
    # Optimization parameters
    parser.add_argument('--min_epochs', type=int, default=1, help='Min number of epochs to perform.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max number of epochs to perform.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size when processing data.')
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    # Permutation invariance parameter
    parser.add_argument('--pi_mode', type=str, default='mean', help='Permutation invariance mode.')
    # GECO-related parameters
    parser.add_argument('--use_GECO', action='store_true', help='Whether to use GECO optimization.')
    parser.add_argument('--kappa', type=float, default=0.05,
                        help='Kappa sensitivity hyperparameter for reconstruction loss.')
    parser.add_argument('--alpha_smoothing', type=float, default=0.99, help='GECO moving average loss.')
    parser.add_argument('--update_every_batch', type=int, default=2, help='When to update GECO hyperparameter.')
    # Diffeomorphic-related parameters
    parser.add_argument("--number_of_time_points", type=int, default=5, help="Scaling and squaring steps")
    parser.add_argument("--downsampling_grid", type=int, default=1, choices=[0, 1, 2],
                        help="Downsampling grid factor (power of 2)")
    parser.add_argument("--deformation_kernel_width", type=int, default=5, help="Deformation kernel size (wrt grid).")
    parser.add_argument('--unclamp_atlas', action='store_true', help='Whether not to clamp (default is on).')
    parser.add_argument('--isometry_constraint', action='store_true', help='Whether isometry layer is activated.')
    # Verbose
    parser.add_argument('--verbose', type=int, default=20, help='Plots every iter')

    args = parser.parse_args()

    run_main(args)

    """
    -Correlation latent : MI / Mean Corr / PLS
    -BVAE no resampling of indices !! How to do that ? 
    
    TODO
    -Permutation invariance module z_t/z_s 
        --> quoi sauver dans les z en sortie destep ?

    Raph Check:
        - A t on des papiers sur permuation invariance fin de l'encoder ? A check aussi
        - Espérance sur les z_s --> une égularisation sur distance entre visites de même patients ? (cf biblio partagée Igor, à vérifier, avec des RNN)

    Bonus : 
        -pre trainig difféo
    """
