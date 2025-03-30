# training/train.py

from keras.callbacks import EarlyStopping

def train_model(model, x_train, y_train, epochs, batch_size, val_split, patience):
    """
    Train the given model using early stopping.
    
    Parameters:
      model: Compiled Keras model.
      x_train: Training inputs.
      y_train: Training targets.
      epochs: Maximum number of epochs.
      batch_size: Batch size.
      val_split: Fraction of training data for validation.
      patience: Patience for early stopping.
      
    Returns:
      Training history.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=val_split, callbacks=[early_stop])
    return history
