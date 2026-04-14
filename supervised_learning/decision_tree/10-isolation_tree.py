#!/usr/bin/env python3
"""Module pour construire un arbre d`isolation aleatoire"""
import numpy as np

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Arbre d`isolation aleatoire pour detecter les outliers en
        mesurant la profondeur a laquelle chaque individu est isole"""
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Retourne la representation visuelle de tout l`arbre
            en partant de la racine"""
        return self.root.__str__()

    def depth(self):
        """Retourne la profondeur maximale de l`arbre"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """only_leaves=False retourne le nombre de noeuds et feuilles
            dans l`arbre, si only_leaves=True retourne uniquement les
            feuilles"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Lance le calcul des bornes depuis la racine"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Retourne la liste de toutes les feuilles
            de l`arbre en partant de la racine"""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Prepare la fonction predict en calculant les bornes et
            les indicateurs de chaque feuille pour pouvoir predire
            efficacement la classe d`un individu"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: (
            np.array([leaf.value for leaf in leaves])[
                np.argmax(np.array([leaf.indicator(A)
                                    for leaf in leaves]), axis=0)])

    def np_extrema(self, arr):
        """Retourne le minimum et le maximum d`un tableau"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choisit aleatoirement une feature et un seuil pour diviser
            le noeud (node) en deux sous-populations"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Cree une feuille enfant dont la valeur est sa profondeur
            car on veut savoir a quelle profondeur l`individu a fini"""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth+1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Cree un noeud enfant du noeud (node) avec sa profondeur
            et sa sous-population (sub_population)"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Construit recursivement les enfants du noeud en splitant
            la population jusqu`a atteindre la profondeur maximale
            ou une population trop petite"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (
            self.explanatory[:, node.feature] > node.threshold
        ) & node.sub_population
        right_population = (
            self.explanatory[:, node.feature] <= node.threshold
        ) & node.sub_population

        # Is left node a leaf ?
        is_left_leaf = (
            np.sum(left_population) <= 1
            or node.depth >= self.max_depth - 1
            or np.sum(left_population) <= 1
            )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            np.sum(right_population) <= 1
            or node.depth >= self.max_depth - 1
            or np.sum(right_population) == 0
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Entraine l`arbre d`isolation sur les donnees sans target
            en construisant l`arbre avec des splits aleatoires"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth : {self.depth()}
    - Number of nodes : {self.count_nodes()}
    - Number of leaves : {self.count_nodes(only_leaves=True)}""")
