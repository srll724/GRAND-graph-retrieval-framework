import tensorflow as tf


def pairwise_euclidean_similarity(x, y):
    s = 2 * tf.matmul(x, y, transpose_b=True)
    diag_x = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    diag_y = tf.reshape(tf.reduce_sum(y ** 2, axis=-1), [1, -1])
    return s - diag_x - diag_y
