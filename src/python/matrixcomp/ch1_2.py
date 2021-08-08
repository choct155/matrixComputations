import tensorflow as tf

node_matrix_a: tf.Tensor = tf.constant([
    [2.0, -1.0, -1.0, 0.0],
    [-1.0, 1.5, 0.0, -0.5],
    [-1.0, 0.0, 1.7, -0.2],
    [0.0, -0.5, -0.2, 1.7]])

node_matrix_b: tf.Tensor = tf.constant([[0.0], [0.0], [3.0], [0.0]])

def solve_for_x(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    node_matrix_x: tf.Tensor = tf.linalg.solve(a, b, adjoint=False)
    return node_matrix_x

print(solve_for_x(node_matrix_a,node_matrix_b))

def residual(a: tf.Tensor, b: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    residual: tf.Tensor = b - tf.matmul(a, x)
    return residual

