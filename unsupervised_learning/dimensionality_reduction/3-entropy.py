#!/usr/bin/env python3
"""Calculates the Shannon entropy and P affinities relative to a data point"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data
    point

    Args:
        Di (numpy.ndarray): shape (n - 1,) containing the pairwise
            distances between a data point and all other points except
            itself
        beta (numpy.ndarray): shape (1,) containing the beta value for
            the Gaussian distribution

    Returns:
        (Hi, Pi):
            Hi (float): the Shannon entropy of the points
            Pi (numpy.ndarray): shape (n - 1,) containing the P affinities
                of the points
    """
    # unnormalized Gaussian weights for each neighbor
    j_weight = np.exp(-Di * beta)
    summ = np.sum(j_weight)
    # P affinities of the point relative to all other points
    Pi = j_weight / summ
    # Shannon entropy of the resulting distribution
    Hi = -np.sum(Pi * np.log2(np.maximum(Pi, 1e-12)))
    return (np.squeeze(Hi), Pi)
