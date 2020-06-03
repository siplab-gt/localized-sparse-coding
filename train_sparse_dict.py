"""
Train sparse dictionary model (Olshausen 1997) with whitened images used in the original paper. This script includes
the option to train with specific compression matrices.

@Filename    train_sparse_dict
@Author      Kion
@Created     5/29/20
"""
import argparse
import time

import numpy as np
import scipy.io
from sklearn.feature_extraction.image import extract_patches_2d

from utils.solvers import FISTA, ADMM
from utils.sparse_mat import dbd_matrix, brm_matrix, hilbert_curve

# PARSE COMMAND LINE ARGUMENTS #
parser = argparse.ArgumentParser(description='Run sparse dictionary learning with compressed images.')
parser.add_argument('-S', '--solver', default='FISTA', choices=['FISTA', 'ADMM'],
                    help="Solver used to find sparse coefficients")
parser.add_argument('-b', '--batch_size', default=100, type=int, help="Batch size")
parser.add_argument('-N', '--dict_count', default=256, type=int, help="Dictionary count")
parser.add_argument('-p', '--patch_size', default=16, type=int, help="Patch size")
parser.add_argument('-R', '--l1_penalty', default=1e-1, type=float, help="L1 regularizer constant")
parser.add_argument('-e', '--num_epochs', default=100, type=int, help="Number of epochs")
parser.add_argument('-T', '--train_samples', default=60000, type=int, help="Number of training samples to use")
parser.add_argument('-V', '--val_samples', default=15000, type=int, help="Number of validation samples to use")
parser.add_argument('-C', '--corr_samples', default=8000, type=int,
                    help="Number of correlation samples to use to recover dictionaries")
parser.add_argument('-c', '--compression', required=True, choices=['none', 'dbd', 'brm'],
                    help="Type of compression to use. None for regular sparse dictionary, dbd for distinct block "
                         "diagonal, brm for banded diagonal.")
parser.add_argument('-j', '--localization', required=True, type=int,
                    help="Degree of localization for compression. J=1 has no localization.")
parser.add_argument('-r', '--compression_ratio', default=.5, type=float, help="Ratio of compression")
parser.add_argument('-l', '--learning_rate', default=0.8, type=float, help="Default initial learning rate")
parser.add_argument('-d', '--decay', default=.975, type=float, help="Default multiplicative learning rate decay")

# PARSE ARGUMENTS #
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
M_tilde = int((patch_size ** 2) * args.compression_ratio)
compression = args.compression

save_suffix = time.strftime("%m-%d-%Y") + "_" + compression + "_J" + str(J) + "_tau" + str(tau)

if __name__ == "__main__":
    # LOAD DATA #
    data_matlab = scipy.io.loadmat('./data/whitened_images.mat')
    images = np.ascontiguousarray(data_matlab['IMAGES'])

    # Extract patches using SciKit-Learn. Out of 10 images, 8 are used for training and 2 are used for validation.
    data_patches = np.moveaxis(extract_patches_2d(images[:, :, :-2], (patch_size, patch_size)), -1, 1). \
        reshape(-1, patch_size, patch_size)
    val_patches = extract_patches_2d(images[:, :, -2], (patch_size, patch_size))

    # Designate patches to use for training, validation, and correlation (only for compressed dictionaries). This
    # step will also normalize the data.
    val_patches = val_patches[np.linspace(1, val_patches.shape[0] - 1, val_samples, dtype=int), :, :]
    val_patches = val_patches / np.linalg.norm(val_patches.reshape(-1, patch_size ** 2), axis=1)[:, None, None]
    print("Shape of validation dataset: {}".format(val_patches.shape))

    train_idx = np.linspace(1, data_patches.shape[0] - 1, train_samples, dtype=int)
    train_patches = data_patches[train_idx, :, :]
    train_patches = train_patches / np.linalg.norm(train_patches.reshape(-1, patch_size ** 2), axis=1)[:, None, None]
    print("Shape of training dataset: {}".format(train_patches.shape))

    if compression != 'none':
        mask = np.ones(data_patches.shape[0], dtype=bool)
        mask[train_idx] = False
        unused_data = data_patches[mask]
        corr_idx = np.linspace(1, unused_data.shape[0] - 1, corr_samples, dtype=int)
        corr_patches = unused_data[corr_idx, :, :]
        corr_patches = corr_patches / np.linalg.norm(corr_patches.reshape(-1, patch_size ** 2), axis=1)[:, None, None]
        print("Shape of correlation dataset: {}".format(corr_patches.shape))

    # Generate Hilbert Curve to arrange vectorized image patches to obey spatial locality (i.e. pixels close in image
    # stay close to one another)
    if J > 1:
        index_ordering = hilbert_curve(J, patch_size)
    else:
        index_ordering = np.arange(patch_size ** 2)

    # INITIALIZE TRAINING PARAMETERS #
    dictionary = np.random.randn(patch_size ** 2, num_dictionaries)
    dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))
    step_size = learning_rate

    # Create compression matrices and compress the dictionary
    if compression == 'dbd':
        compression_matrix = dbd_matrix(J, M_tilde, patch_size ** 2)
        compressed_dictionary = compression_matrix @ dictionary
    elif compression == 'brm':
        compression_matrix = brm_matrix(M_tilde // J, M_tilde, patch_size ** 2)
        compressed_dictionary = compression_matrix @ dictionary

    # Initialize empty arrays for tracking learning data
    dictionary_saved = np.zeros((num_epochs, *dictionary.shape))
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    train_time = np.zeros(num_epochs)

    # TRAIN MODEL #
    init_time = time.time()
    for j in range(num_epochs):
        epoch_loss = np.zeros(train_patches.shape[0] // batch_size)
        # Shuffle training data-set
        np.random.shuffle(train_patches)
        for i in range(train_patches.shape[0] // batch_size):
            patches = train_patches[i * batch_size:(i + 1) * batch_size].reshape(batch_size, -1).T
            # Re-order vectorized patches to obey spatial locality
            patches = np.squeeze(patches[index_ordering, :])

            # Create single variable that can be used irrespective of compression matrix.
            if compression == 'none':
                infer_dictionary = dictionary
                infer_patches = patches
            else:
                infer_dictionary = compressed_dictionary
                # Compress patches and then normalize
                infer_patches = compression_matrix @ patches
                # TODO: Determine if this breaks statistical guarantees for convergence
                infer_patches /= np.linalg.norm(infer_patches, axis=0)

            # Infer coefficients
            if solver == "FISTA":
                b = FISTA(infer_dictionary, infer_patches, tau=tau)
            elif solver == "ADMM":
                b = ADMM(infer_dictionary, infer_patches, tau=tau)

            # Take gradient step on dictionaries
            generated_patch = infer_dictionary @ b
            residual = infer_patches - generated_patch
            step = residual @ b.T
            infer_dictionary += step_size * step

            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            infer_dictionary /= np.sqrt(np.sum(infer_dictionary ** 2, axis=0))
            # Calculate loss after gradient step
            epoch_loss[i] = 0.5 * np.sum((infer_patches - infer_dictionary @ b) ** 2) + tau * np.sum(np.abs(b))

        if compression != 'none':
            C_sr = []
            C_rr = []
            for i in range(corr_patches.shape[0] // batch_size):
                # Load next batch of correlation patches
                patches = corr_patches[i * batch_size:(i + 1) * batch_size].reshape(batch_size, -1).T
                # Re-order vectorized patches to obey spatial locality
                patches = np.squeeze(patches[index_ordering, :])
                # Compress patches
                compressed_patches = compression_matrix @ patches

                # Infer coefficients
                if solver == "FISTA":
                    b = FISTA(compressed_dictionary, compressed_patches, tau=tau)
                elif solver == "ADMM":
                    b = ADMM(compressed_dictionary, compressed_patches, tau=tau)

                # Append to list used for reconstructing dictionaries
                C_sr.append(patches @ b.T)
                C_rr.append(b @ b.T)

            # Take average over all samples
            C_sr = np.mean(np.array(C_sr), axis=0) / batch_size
            C_rr = np.mean(np.array(C_rr), axis=0) / batch_size
            # Reconstruct dictionaries
            dictionary_rm = C_sr @ np.linalg.pinv(C_rr)
            dictionary_rm /= np.sqrt(np.sum(dictionary_rm ** 2, axis=0))

        # Test reconstructed or uncompressed dictionary on validation data-set
        epoch_val_loss = np.zeros(val_patches.shape[0] // batch_size)
        for i in range(val_patches.shape[0] // batch_size):
            # Load next batch of validation patches
            patches = val_patches[i * batch_size:(i + 1) * batch_size].reshape(batch_size, -1).T
            # Re-order vectorized patches to obey spatial locality
            patches = np.squeeze(patches[index_ordering, :])
            
            if compression == 'none':
                infer_dictionary = dictionary
            else:
                infer_dictionary = dictionary_rm

            # Infer coefficients
            if solver == "FISTA":
                b_hat = FISTA(infer_dictionary, patches, tau=tau)
            elif solver == "ADMM":
                b_hat = ADMM(infer_dictionary, patches, tau=tau)

            # Compute and save loss
            epoch_val_loss[i] = 0.5 * np.sum((patches - infer_dictionary @ b_hat) ** 2) + tau * np.sum(np.abs(b_hat))

        # Decay step-size
        step_size = step_size * decay

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j] = np.mean(epoch_loss)
        val_loss[j] = np.mean(epoch_val_loss)
        dictionary_saved[j] = infer_dictionary
        np.savez_compressed('results/traindata_' + save_suffix,
                            phi=dictionary_saved, time=train_time,
                            train=train_loss, val=val_loss)
        print("Epoch {} of {}, Avg Train Loss = {:.4f}, Avg Val Loss = {:.4f}, Time = {:.0f} secs".format(j + 1,
                                                                                                          num_epochs,
                                                                                                          train_loss[j],
                                                                                                          val_loss[j],
                                                                                                          epoch_time))
