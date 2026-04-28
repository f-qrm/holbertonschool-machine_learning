#!/usr/bin/env python3
""" Module that trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """ Function that trains a model using mini-batch gradient descent """
    callbacks = []
    if early_stopping and validation_data:
        erl_stp = K.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience)
        callbacks.append(erl_stp)
    if learning_rate_decay and validation_data:
        def schedule(epoch):
            """Function that computes the learning rate for each epoch"""
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(schedule, verbose=1)
        callbacks.append(lr_decay)
    if save_best and validation_data:
        new = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=True,
            monitor='val_loss'
        )
        callbacks.append(new)
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
