import unittest
import os
import argparse
import numpy as np
from main import run_main


class test_run_longvae(unittest.TestCase):

    def base_arguments(self):
        # Parser
        parser = argparse.ArgumentParser(description='Longitudinal Unsupervised | No-GECO | LIGHTNING VERSION.')
        # General action parameters
        # parser.add_argument('--data_dir', type=str, default='Data', help='Data directory root.')
        parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
        parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
        parser.add_argument('--num_threads', type=int, default=36,
                            help='Number of threads to use if cuda not available')
        parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
        parser.add_argument('--num_workers', type=int, default=0, help='Number of workers.')
        # Dataset parameters
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
        parser.add_argument('--decoder_last_activation', type=str, default='identity',
                            help='decoder last_function name.')
        parser.add_argument('--one_encoder', action='store_true', help='Whether to use only 1 encoder')
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
        parser.add_argument('--cliplogvar_min', type=float, default=float(-2 * 2 * np.log(10)),
                            help='10**min clip variance.')
        parser.add_argument('--cliplogvar_max', type=float, default=float(3 * 2 * np.log(10)),
                            help='10**max clip variance.')
        parser.add_argument("--tol", type=float, default=float(1e-10), help="Numerical tolerance.")
        # Optimization parameters
        parser.add_argument('--min_epochs', type=int, default=1, help='Min number of epochs to perform.')
        parser.add_argument('--max_epochs', type=int, default=10, help='Max number of epochs to perform.')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size when processing data.')
        parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
        # Permutation invariance parameter
        parser.add_argument('--pi_mode', type=str, default='set', help='Permutation invariance mode.')
        # GECO-related parameters
        parser.add_argument('--use_GECO', action='store_true', help='Whether to use GECO optimization.')
        parser.add_argument('--kappa', type=float, default=0.05,
                            help='Kappa sensitivity hyperparameter for reconstruction loss.')
        parser.add_argument('--alpha_smoothing', type=float, default=0.99, help='GECO moving average loss.')
        parser.add_argument('--update_every_batch', type=int, default=2, help='When to update GECO hyperparameter.')
        # Diffeomorphic-related parameters
        parser.add_argument("--number_of_time_points", type=int, default=5, help="Scaling and squaring steps")
        parser.add_argument("--downsampling_grid", type=int, default=0, choices=[0, 1, 2],
                            help="Downsampling grid factor (power of 2)")
        parser.add_argument("--deformation_kernel_width", type=int, default=5,
                            help="Deformation kernel size (wrt grid).")
        parser.add_argument('--unclamp_atlas', action='store_true', help='Whether not to clamp (default is on).')
        parser.add_argument('--isometry_constraint', action='store_true', help='Whether isometry layer is activated.')
        # Verbose
        parser.add_argument('--verbose', type=int, default=1, help='Do all plots')

        args = parser.parse_args()

        MODEL_NAME = "LongVAE"
        NUM_VISITS = 200
        MAX_EPOCHS = 1
        VERBOSE = 1
        LATENT_DIMENSION = 5
        PI_MODE = "mean"
        args.model_name = MODEL_NAME
        args.num_visits = NUM_VISITS
        args.max_epochs = MAX_EPOCHS
        args.verbose = VERBOSE
        args.latent_dimension = LATENT_DIMENSION
        args.pi_mode = PI_MODE
        args.cuda = True
        return args

    def test_run_adni_MRI_2D_64(self):
        args = self.base_arguments()
        args.dataset = "ADNI_MRI"
        args.dataset_version = "path_2d_midaxial_64"
        run_main(args)
        return 0

    def test_run_adni_MRI_3D_64(self):
        args = self.base_arguments()
        args.dataset = "ADNI_MRI"
        args.dataset_version = "path_3d_64"
        run_main(args)
        return 0

    def test_run_PPMI_DAT_2D_64(self):
        args = self.base_arguments()
        args.dataset = "PPMI_DAT"
        args.dataset_version = "path_sliced41_reshaped"
        run_main(args)
        return 0

    def test_run_adni_1D(self):
        args = self.base_arguments()
        args.dataset = "ADNI_COG"
        args.dataset_version = "2d_midaxial"
        run_main(args)
        return 0