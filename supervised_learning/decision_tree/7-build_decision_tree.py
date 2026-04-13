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

    def get_leaves_below(self):
        """Retourne la liste de toutes les feuilles
            en dessous de ce noeud (toute sa descendance)"""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Propage les bornes (lower et upper) de haut en bas dans l`arbre.
            Chaque enfant herite des bornes du parent. L`enfant gauche recoit
            le seuil comme borne minimale (lower) car on y arrive quand
            feature >= threshold. L`enfant droit recoit le seuil comme borne
            maximale (upper) car on y arrive quand feature < threshold"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child == self.left_child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                child.lower[self.feature] = self.threshold
            else:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Cree une fonction indicator qui verifie si un individu
            appartient a la region de ce noeud en verifiant que
            toutes ses features sont dans les bornes lower et upper"""
        def is_large_enough(x):
            """Retourne True pour chaque individu dont toutes les
                features sont superieures aux bornes minimales (lower)"""
            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """Retourne True pour chaque individu dont toutes les
                features sont inferieures ou egales aux bornes
                maximales (upper)"""
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                          is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Retourne la prediction en allant a gauche si la feature
            est superieure au seuil, sinon a droite"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def get_leaves_below(self):
        """Retourne les feuilles en forme de liste afin de
            pouvoir mettre tous les feuilles dans une liste"""
        return [self]

    def update_bounds_below(self):
        """Recoit ses bornes du parent et s`arrete la car
            une feuille n`as pas d`enfants"""
        pass

    def pred(self, x):
        """Retourne la valeur de la feuille comme prediction finale"""
        return self.value


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
        """only_leaves=False retourne le nombre de noeuds et feuilles
            dans l`arbre, si only_leaves=True retourne uniquement les
            feuilles"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Retourne la representation visuelle de tout l`arbre
            en partant de la racine"""
        return self.root.__str__()

    def get_leaves(self):
        """Retourne la liste de toutes les feuilles
            de l`arbre en partant de la racine"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Lance le calcul des bornes depuis la racine"""
        self.root.update_bounds_below()

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

    def pred(self, x):
        """Retourne la prediction pour un individu en parcourant
            l`arbre depuis la racine"""
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Entraine l`arbre sur les donnees (explanatory) et les cibles
            (target) en choisissant le critere de split et en construisant
            l`arbre noeud par noeud"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {
                                    self.accuracy(self.explanatory,
                                                  self.target)
                                }""")

    def np_extrema(self, arr):
        """Retourne le minimum et le maximum d`un tableau (arr)"""
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

    def fit_node(self, node):
        """Construit recursivement les enfants du noeud (node) en
            splitant la population jusqu`a atteindre une condition d`arret"""
        node.feature, node.threshold = self.split_criterion(node)
        left_population = (
            self.explanatory[:, node.feature] > node.threshold
        ) & node.sub_population
        right_population = (
            self.explanatory[:, node.feature] <= node.threshold
        ) & node.sub_population
        # Is left node a leaf ?
        is_left_leaf = (
            np.sum(left_population) < self.min_pop
            or node.depth + 1 >= self.max_depth
            or len(np.unique(self.target[left_population])) == 1
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
        # Is right node a leaf ?
        is_right_leaf = (
            np.sum(right_population) < self.min_pop
            or node.depth + 1 >= self.max_depth
            or len(np.unique(self.target[right_population])) == 1
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Cree une feuille enfant du noeud (node) avec la classe
            la plus representee parmi les individus (sub_population)"""
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Cree un noeud enfant du noeud (node) avec sa profondeur
            et sa sous-population (sub_population)"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calcule la proportion de bonnes predictions sur les donnees
            de test (test_explanatory) par rapport aux vraies cibles
            (test_target)"""
        return (np.sum(np.equal(self.predict(test_explanatory), test_target))
                / test_target.size)
