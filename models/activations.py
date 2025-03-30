# models/activations.py

import tensorflow as tf
from keras import layers

class CustomTanh(layers.Layer):
    """Custom activation layer computing tanh(alpha * x) with trainable alpha."""
    def __init__(self, initial_alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_alpha = initial_alpha

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_alpha),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_alpha": self.initial_alpha
        })
        return config
    
class Alsing(layers.Layer):
    def __init__(self, initial_beta=1.0, initial_gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_beta = initial_beta
        self.initial_gamma = initial_gamma

    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_beta),
            trainable=True
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_gamma),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # Implements f(x) = (gamma + (1 - gamma)/(1 + exp(-beta * x))) * x
        return (self.gamma + (1 - self.gamma) / (1 + tf.exp(-self.beta * inputs))) * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_beta": self.initial_beta,
            "initial_gamma": self.initial_gamma
        })
        return config
