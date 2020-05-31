"""
Implementation of different sparse matrices used for compressed dictionary learning

@Filename    sparse_mat
@Author      Kion 
@Created     5/30/20
"""
import numpy as np
import scipy
from PIL import Image


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


def bd_matrix(bands, M, N):
    """
    Returns a sparse banded diagonal (BD) matrix with zero-mean Gaussian entries with variance 1/M
    :param bands: Number of bands from the diagonal to have non-zero
    :param M: Rows in resulting matrix
    :param N: Columns in resulting matrix
    :return: Numpy array containing M by N sparse BD matrix
    """
    comp_block_diag = np.random.randn(M, N) * (1 / np.sqrt(M))
    diag_square = np.zeros((M, M))
    for i in range(M):
        for j in range(-bands // 2, bands // 2):
            diag_square[i, (i + j) % M] = 1
    diag_image = Image.fromarray(diag_square)
    diag_band = np.array(diag_image.resize((M, N), resample=Image.NEAREST))
    return diag_band * comp_block_diag
