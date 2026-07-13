#!/usr/bin/env python3
""" Module that trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent, with optional
    early stopping based on validation loss.

        Args:
            network (K.Model): The model to train.
            data (numpy.ndarray): Input data of shape (m, nx).
            labels (numpy.ndarray): One-hot labels of shape (m, classes).
            batch_size (int): Size of the batch used for mini-batch
                gradient descent.
            epochs (int): Number of passes through data for mini-batch
                gradient descent.
            validation_data (tuple, optional): Data to validate the model
                with, as (data, labels).
            early_stopping (bool): Whether early stopping should be used.
                Only takes effect if validation_data is provided.
            patience (int): Number of epochs with no improvement in
                val_loss to wait before stopping early.
            verbose (bool): Whether output should be printed during
                training.
            shuffle (bool): Whether to shuffle the batches every epoch.

        Returns:
            History: The History object generated after training.
    """
    callbacks = []
    if early_stopping and validation_data:
        # monitor='val_loss' requires validation_data; early stopping is
        # only wired in when both conditions hold, hence the guard above.
        erl_stp = K.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience)
        callbacks.append(erl_stp)
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
