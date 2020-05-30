"""
Train sparse dictionary model (Olshausen 1997) with whitened images used in the original paper. This script includes
the option to train with specific compression matrices.

@Filename    train_sparse_dict
@Author      Kion 
@Created     5/29/20
"""
import argparse

import numpy as np
import scipy
# PARSE COMMAND LINE ARGUMENTS #
from sklearn.feature_extraction.image import extract_patches_2d

parser = argparse.ArgumentParser(description='Run sparse dictionary learning with compressed images.')
parser.add_argument('-S', '--solver', default='FISTA', choices=['FISTA', 'ADMM'],
                    help="Solver used to find sparse coefficients")
parser.add_argument('-b', '--batch_size', default=100, type=int, help="Batch size")
parser.add_argument('-N', '--dict_count', default=169, type=int, help="Dictionary count")
parser.add_argument('-p', '--patch_size', default=16, type=int, help="Patch size")
parser.add_argument('-r', '--l1_penalty', default=4e-2, type=float, help="L1 regularizer constant")
parser.add_argument('-e', '--num_epochs', default=60, type=int, help="Number of epochs")
parser.add_argument('-T', '--train_samples', default=40000, type=int, help="Number of training samples to use")
parser.add_argument('-V', '--val_samples', default=10000, type=int, help="Number of validation samples to use")
parser.add_argument('-C', '--corr_samples', default=1280, type=int,
                    help="Number of correlation samples to use to recover dictionaries")
parser.add_argument('-c', '--compression', required=True, choices=['none', 'dbd', 'bd'],
                    help="Type of compression to use. None for regular sparse dictionary, dbd for distinct block "
                         "diagonal, bd for banded diagonal.")
parser.add_argument('-j', '--localization', required=True,
                    help="Degree of localization for compression. J=1 is not localization.")
parser.add_argument('-l', '--learning_rate', default=.4, type=float, help="Default initial learning rate")
parser.add_argument('-d', '--decay', default=.985, type=float, help="Default multiplicative learning rate decay")

args = parser.parse_args()
solver = args.solver
batch_size = args.batch_size
num_dictionaries = args.dict_count
patch_size = args.patch_size
tau = args.l1_penalty
num_epochs = args.num_epochs
train_samples = args.train_samples
val_samples = args.val_samples
corr_samples = args.corr_samples
learning_rate = args.learning_rate
decay = args.decay
J = args.localization
compression = args.compression

if __name__ == "__main__":
    # LOAD DATA #
    data_matlab = scipy.io.loadmat('./data/whitened_images.mat')
    images = np.ascontiguousarray(data_matlab['IMAGES'])

    # Extract patches using SciKit-Learn. Out of 10 images, 8 are used for training and 2 are used for validation
    data_patches = np.moveaxis(extract_patches_2d(images[:, :, :-2], (patch_size, patch_size)), -1, 1). \
        reshape(-1, patch_size, patch_size)
    val_patches = extract_patches_2d(images[:, :, -2], (patch_size, patch_size))

    # Designate patches to use for
