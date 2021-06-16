import tensorflow as tf

def testRandomMatrices() -> tf.Tensor:
    a: tf.Tensor = tf.random.normal((3, 4), 0.0, 1.0)
    x: tf.Tensor = tf.random.normal((4, 1), 0.0, 1.0)
    b: tf.Tensor = tf.linalg.matmul(a, x)
    return b


