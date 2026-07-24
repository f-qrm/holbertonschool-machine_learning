#!/usr/bin/env python3
"""Performs K-means on a dataset using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters

    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index
        of the cluster in X that each data point belongs to
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
