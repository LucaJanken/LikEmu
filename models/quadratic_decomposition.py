import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Layer

class QuadraticDecomposition(Layer):
    def __init__(self, N, **kwargs):
        super(QuadraticDecomposition, self).__init__(**kwargs)
        self.N = N

    def call(self, inputs):
        # Unpack inputs: x_input and params.
        x_input, params = inputs
        # f(x₀)
        f0 = params[:, 0]
        # Local minimizer x₀ (shape: batch x N)
        x0 = params[:, 1:1+self.N]
        # Flattened entries for the Cholesky factor (for lower-triangular matrix)
        L_flat = params[:, 1+self.N:]
        # Reconstruct lower-triangular matrix.
        L_lower = tfp.math.fill_triangular(L_flat)
        # Ensure covariance matrix is positive definite.
        L_lower = tf.linalg.set_diag(L_lower, tf.nn.softplus(tf.linalg.diag_part(L_lower)))
        # Transpose to get L^T.
        L_T = tf.transpose(L_lower, perm=[0, 2, 1])
        # Compute difference.
        diff = tf.expand_dims(x_input - x0, axis=-1)
        # Compute quadratic term: ||L^T (x - x₀)||².
        LTx = tf.matmul(L_T, diff)
        quad_term = tf.reduce_sum(tf.square(LTx), axis=[1,2])
        return f0 + quad_term

    def get_config(self):
        config = super(QuadraticDecomposition, self).get_config()
        config.update({"N": self.N})
        return config
