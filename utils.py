import numpy as np
import tensorflow as tf
from keras import backend as K

def batch_matmul(A, B, transpose_a=False, transpose_b=False):
    # adopted from here: https://github.com/tensorflow/tensorflow/issues/216
    '''Batch support for matrix matrix product.

    Args:
        A: General matrix of size (A_Batch, M, X).
        B: General matrix of size (B_Batch, X, N).
        transpose_a: Whether A is transposed (A_Batch, X, M).
        transpose_b: Whether B is transposed (B_Batch, N, X).

    Returns:
        The result of multiplying A with B (A_Batch, B_Batch, M, N).
        Works more efficiently if B_Batch is empty.
    '''
    Andim = len(A.shape)
    Bndim = len(B.shape)
    if Andim == Bndim:
        return K.tf.matmul(A, B, transpose_a=transpose_a,
                         transpose_b=transpose_b)  # faster than tensordot
    with K.tf.name_scope('matmul'):
        a_index = Andim - (2 if transpose_a else 1)
        b_index = Bndim - (1 if transpose_b else 2)
        AB = K.tf.tensordot(A, B, axes=[a_index, b_index])
        if Bndim > 2:  # only if B is batched, rearrange the axes
            A_Batch = np.arange(Andim - 2)
            M = len(A_Batch)
            B_Batch = (M + 1) + np.arange(Bndim - 2)
            N = (M + 1) + len(B_Batch)
            perm = np.concatenate((A_Batch, B_Batch, [M, N]))
            AB = K.tf.transpose(AB, perm)
    return AB