#!/usr/bin/env python3
"""Perform semantic search on a corpus of documents."""
# Pour lister les fichiers du dossier corpus
import os
# Pour le produit scalaire, les normes et l'argmax
import numpy as np
# Pour charger le Universal Sentence Encoder pre-entraine
import tensorflow_hub as hub

# USE transforme une phrase ou un document entier en un vecteur de
# taille fixe : deux textes de sens proche donnent des vecteurs proches,
# c'est ce qui permet de comparer le sens et pas juste les mots
model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5")


def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.

    Args:
        corpus_path (str): the path to the corpus of reference
            documents on which to perform semantic search.
        sentence (str): the sentence from which to perform semantic
            search.

    Returns:
        str: the reference text of the document most similar to
            sentence.
    """
    # Liste des textes, dans le meme ordre que leurs embeddings
    documents = []
    # Parcourt tous les fichiers du dossier pour construire le corpus
    for filename in os.listdir(corpus_path):
        # On ne garde que les articles markdown, pas d'autres fichiers
        if filename.endswith(".md"):
            # os.listdir ne renvoie que le nom, il faut le chemin complet
            file_path = corpus_path + "/" + filename
            # with garantit la fermeture du fichier meme en cas d'erreur
            with open(file_path) as f:
                # Le document entier sert de texte de reference
                ref = f.read()
            # Garde le texte pour pouvoir le renvoyer tel quel a la fin
            documents.append(ref)
    # La phrase est ajoutee a la fin pour tout encoder en un seul
    # appel au modele, plus rapide que deux appels separes
    corpus = documents + [sentence]
    # Un vecteur par texte, dans le meme ordre que corpus
    embeddings = model(corpus)
    # La phrase etant le dernier element, son vecteur est le dernier
    ques = embeddings[-1]
    # Un score de similarite par document
    similarites = []
    # [:-1] exclut la phrase elle-meme, qui serait parfaitement
    # similaire a elle-meme
    for doc_embedding in embeddings[:-1]:
        # Similarite cosinus : compare la direction des vecteurs (le
        # sens) sans etre influencee par leur longueur
        similarity = np.dot(ques, doc_embedding) / (
            np.linalg.norm(ques) * (np.linalg.norm(doc_embedding)))
        # Garde le score a la meme position que le document
        similarites.append(similarity)
    # Le score le plus eleve correspond au document le plus proche
    best_index = np.argmax(similarites)
    # Meme indice dans documents grace a l'ordre conserve
    return documents[best_index]
