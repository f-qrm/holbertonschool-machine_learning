#!/usr/bin/env python3
"""Module pour construire une foret aleatoire de decision"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Une foret aleatoire qui combine plusieurs arbres de decision
        pour faire des predictions par vote majoritaire"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Retourne la classe la plus votee par tous les arbres
            pour chaque individu dans explanatory"""
        predictions = []
        for pred in self.numpy_preds:
            predictions.append(pred(explanatory))
        predictions = np.array(predictions)
        return np.array([np.bincount(predictions[:, i]).argmax()
                        for i in range(predictions.shape[1])])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Entraine n_trees arbres de decision sur les donnees
            et stocke leurs fonctions de prediction"""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                                        self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Calcule la proportion de bonnes predictions sur
            les donnees de test"""
        return np.sum(np.equal(self.predict(test_explanatory),
                      test_target))/test_target.size
