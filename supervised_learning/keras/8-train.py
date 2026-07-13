#!/usr/bin/env python3
""" Module that trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent, with optional
    early stopping, learning rate decay, and checkpointing of the best
    model seen during training.

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
            learning_rate_decay (bool): Whether learning rate decay
                should be used. Only takes effect if validation_data is
                provided.
            alpha (float): The initial learning rate.
            decay_rate (float): The decay rate used for inverse time
                decay of the learning rate.
            save_best (bool): Whether to save the model with the best
                validation loss seen during training.
            filepath (str, optional): Path to save the best model to.
            verbose (bool): Whether output should be printed during
                training.
            shuffle (bool): Whether to shuffle the batches every epoch.

        Returns:
            History: The History object generated after training.
    """
    callbacks = []
    if early_stopping and validation_data:
        erl_stp = K.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience)
        callbacks.append(erl_stp)
    if learning_rate_decay and validation_data:
        def schedule(epoch):
            """Compute the inverse time decayed learning rate for the
            given epoch.
            """
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(schedule, verbose=1)
        callbacks.append(lr_decay)
    if save_best and validation_data:
        # save_best_only keeps only the checkpoint with the lowest
        # val_loss seen so far, so training can be stopped early without
        # losing the best-performing weights.
        best_checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=True,
            monitor='val_loss'
        )
        callbacks.append(best_checkpoint)
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
