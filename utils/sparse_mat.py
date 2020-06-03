"""
Implementation of different sparse matrices used for compressed dictionary learning

@Filename    sparse_mat
@Author      Kion 
@Created     5/30/20
"""
import numpy as np
import scipy
import scipy.linalg
from PIL import Image


def hilbert_curve(num_blocks, patch_size):
    """
    Returns a hilbert curve which can be used to ensure that a vectorized image patch maintains spatial locality.
    https://blogs.mathworks.com/steve/2012/01/25/generating-hilbert-curves/
    :param num_blocks: Number of blocks used dto
    :param patch_size:
    """
    order = int(np.log(patch_size ** 2) / np.log(num_blocks))
    a = 1 + 1j
    b = 1 - 1j
    z = np.expand_dims(np.array([0]), -1)
    for k in range(order):
        w = 1j * np.conj(z)
        z = np.concatenate((np.concatenate((w - a, z - b), axis=1), np.concatenate((z + a, b - w), axis=1)), axis=1) / 2

    col = np.real(z) * patch_size / 2 + patch_size / 2 + 0.5
    row = np.imag(z) * patch_size / 2 + patch_size / 2 + 0.5
    indices = np.array((col - 1) * patch_size + row, dtype=int) - 1


def dbd_matrix(num_blocks, M, N):
    """
    Returns sparse distinct block diagonal (DBD) matrix with each element drawn from a Gaussian with zero-mean and
    variance 1/M
    :param num_blocks: Number of distinct blocks
    :param M: Number of rows in resulting DBD matrix (each block will have M / num_blocks)
    :param N: Number of columns in resulting DBD matrix (each block will have N / num_blocks)
    :return: Numpy array containing DBD matrix, dimensions M by N
    """
    comp_matrices = np.random.randn(num_blocks, int(M / num_blocks), int(N / num_blocks)) * (
            1 / np.sqrt(M / num_blocks))
    dbd_mat = scipy.linalg.block_diag(*comp_matrices)
    return dbd_mat


def brm_matrix(bands, M, N, wrap=False):
    """
    Returns a banded random matrix (BRM) with zero-mean Gaussian entries with variance 1/M
    :param bands: Number of bands from the diagonal to have non-zero
    :param M: Rows in resulting matrix
    :param N: Columns in resulting matrix
    :param wrap: Whether the banded-diagonal allows for wrapping
    :return: Numpy array containing M by N sparse BRM matrix
    """
    comp_block_diag = np.random.randn(M, N) * (1 / np.sqrt(M))
    diag_square = np.zeros((M, M))
    for i in range(M):
        for j in range(-bands // 2, bands // 2):
            if not wrap and (i + j < 0 or i + j >= M):
                continue
            diag_square[i, (i + j) % M] = 1
    diag_image = Image.fromarray(diag_square)
    diag_band = np.array(diag_image.resize((M, N), resample=Image.NEAREST)).T
    return diag_band * comp_block_diag
