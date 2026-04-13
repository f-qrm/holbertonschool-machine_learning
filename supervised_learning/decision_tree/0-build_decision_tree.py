#!/usr/bin/env python3
"""Module pour construire un arbre de decision"""
import numpy as np


class Node:
    """ C`est un noed interne de l`arbre, un noeud qui pose une question et
        bifurque a gauche ou a droite"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """On appelle recursivement la meme methode sur chaque enfant
            gauche et droit, jusqu`a atteindre une feuille qui retourne
            sa propre profondeur. Ensuite max remonte le plus grand nombre
            pour trouver la profondeur maximale de l`arbre"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())


class Leaf(Node):
    """Noeud terminal, il ne pose plus de question, il retourne une valeur
        (la pred. finale)"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Trouver la profondeur le plus eleve parmi tous les noeuds de
            l`arbre"""
        return self.depth


class Decision_Tree():
    """Conteneur principale qui gere l`arbre entier"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Retourne la profondeur maximale de l`arbre"""
        return self.root.max_depth_below()
