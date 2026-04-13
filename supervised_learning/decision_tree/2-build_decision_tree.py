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

    def count_nodes_below(self, only_leaves=False):
        """Si il y a que des feuilles on compte uniquement les feuilles,
            sinon on compte le noeud lui meme (1) plus tous ses descendants
            (noeuds et)"""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        else:
            return (1 + self.left_child.count_nodes_below() +
                    self.right_child.count_nodes_below())

    def left_child_add_prefix(self, text):
        """Ajoute le prefixe +-- et | pour construire
            l`architecture visuelle de l`enfant gauche"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Ajoute le prefixe +-- et des espaces pour construire
            l`architecture visuelle de l`enfant droit"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            if x:
                new_text += ("       "+x)+"\n"
        return (new_text)

    def __str__(self):
        """Retourne la representation visuelle du noeud et
            de tous ses descendants sous forme d`arbre"""
        if self.is_root:
            text = (
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
            )
            text += self.left_child_add_prefix(self.left_child.__str__())
            text += self.right_child_add_prefix(self.right_child.__str__())
            return text
        else:
            text = (
                f"-> node [feature={self.feature},"
                f" threshold={self.threshold}]\n"
            )
            text += self.left_child_add_prefix(self.left_child.__str__())
            text += self.right_child_add_prefix(self.right_child.__str__())
            return text


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

    def count_nodes_below(self, only_leaves=False):
        """Une feuilles retourne toujours 1 car elle est elle meme un
            seule noeud"""
        return 1

    def __str__(self):
        """Retourne la representation visuelle de la feuille
            avec sa valeur"""
        return (f"-> leaf [value={self.value}]")


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

    def count_nodes(self, only_leaves=False):
        """only_leaves=False retourne le nombre de noeudds et feuilles
            dans l`arbre, si only_leaves=True retourne uniquement les
            feuilles"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Retourne la representation visuelle de tout l`arbre
            en partant de la racine"""
        return self.root.__str__()
