import tensorflow as tf
from keras.layers import Dense, Activation, Input
from keras.models import Model
from models.activations import CustomTanh, Alsing
from models.quadratic_decomposition import QuadraticDecomposition  

def build_emulation_model(N, num_hidden_layers, neurons, activation, use_gaussian_decomposition, learning_rate):
    """
    Build the neural network model for likelihood emulation.
    
    Parameters:
      N: Dimensionality of the input.
      num_hidden_layers: Number of hidden layers.
      neurons: Number of neurons in each hidden layer.
      activation: A string specifying the activation function.
                  If "custom_tanh", the CustomTanhActivation is used.
                  Otherwise, the Keras Activation layer is used.
      use_gaussian_decomposition: If True, the network outputs a decomposition with
          1 + N + N(N+1)/2 parameters corresponding to f(x₀), x₀, and the flattened Cholesky factor.
          If False, the network directly emulates the target value.
    
    Returns:
      A compiled Keras Model.
    """
    inp = Input(shape=(N,))
    x = inp

    def apply_activation(x):
        if activation == "custom_tanh":
            return CustomTanh()(x)
        elif activation == "alsing":
            return Alsing()(x)
        else:
            return Activation(activation)(x)

    for _ in range(num_hidden_layers):
        x = Dense(neurons)(x)
        x = apply_activation(x)
    
    if use_gaussian_decomposition:
        output_dim = 1 + N + (N * (N + 1)) // 2
        params = Dense(output_dim)(x)
        # Use our custom QuadraticDecomposition layer.
        f_pred = QuadraticDecomposition(N)([inp, params])
    else:
        f_pred = Dense(1)(x)
    
    model = Model(inputs=inp, outputs=f_pred)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model
