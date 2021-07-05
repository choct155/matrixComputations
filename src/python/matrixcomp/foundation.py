import tensorflow as tf
import numpy as np
from matrixcomp.diag import timed


def rand_sq_mat_vec_mult(dim: int) -> tf.Tensor:
    a: tf.Tensor = tf.random.normal(shape=(dim, dim))
    x: tf.Tensor = tf.random.normal(shape=(dim, 1))
    b: tf.Tensor = tf.linalg.matmul(a, x)
    return b


def rand_sq_mat_vec_mult_py(dim: int) -> tf.Tensor:
    a: tf.Tensor = tf.random.normal(shape=(dim, dim))
    x: tf.Tensor = tf.random.normal(shape=(dim, 1))

    b: np.ndarray = np.zeros(shape=(dim, 1))
    for col in range(dim):
        for row in range(dim):
            b[row] += a[row, col] * x[col]

    return tf.constant(b)


def rand_sq_mat_mat_mult(dim: int) -> tf.Tensor:
    a: tf.Tensor = tf.random.normal(shape=(dim, dim))
    x: tf.Tensor = tf.random.normal(shape=(dim, dim))
    b: tf.Tensor = tf.linalg.matmul(a, x)
    return b

