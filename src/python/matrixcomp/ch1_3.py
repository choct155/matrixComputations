import tensorflow as tf

# 1.3.12
# Write a nonrecursive algorithim that implements column-oriented forward substitution


def triangular_column_forward(x: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    rows, columns = tf.shape(x)
    for i in range(1, rows+1):
        y[i] = b[i]/x[i,i]
        for j in range(1, i-1):
            b[i] = b[i] - x[i, j]*y[i]


