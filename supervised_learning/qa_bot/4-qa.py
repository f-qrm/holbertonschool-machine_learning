#!/usr/bin/env python3
"""Answer questions from multiple reference texts."""
# Pour lister les fichiers du dossier corpus
import os
# Pour le produit scalaire, les normes et l'argmax
import numpy as np
# Pour charger le Universal Sentence Encoder pre-entraine
import tensorflow_hub as hub
# Import dynamique car le nom du fichier commence par un chiffre ;
# renomme en find_answer pour ne pas entrer en conflit avec la
# fonction question_answer definie dans ce fichier
find_answer = __import__('0-qa').question_answer

# USE transforme un texte en vecteur de sens, ce qui permet de choisir
# le document le plus pertinent avant d'interroger BERT
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


def question_answer(corpus_path):
    """
    Answer questions from multiple reference texts until the user
    exits.

    Args:
        corpus_path (str): the path to the corpus of reference
            documents.

    Returns:
        None
    """
    # Variantes courantes pour arreter la boucle, comparees en
    # minuscules pour ne pas dependre de la casse saisie
    exit_words = ('exit', 'quit', 'goodbye', 'bye')
    # Boucle infinie : seul un mot de sortie arrete la conversation
    while True:
        # input() bloque jusqu'a ce que l'utilisateur tape une ligne
        user_input = input("Q: ")
        # Verifie la sortie avant tout calcul, pour ne pas lancer USE
        # et BERT inutilement sur "exit"
        if user_input.lower() in exit_words:
            # Message de fin demande par le sujet
            print("A: Goodbye")
            # Sort de la boucle, ce qui termine la fonction
            break
        # BERT ne peut lire qu'un seul contexte a la fois : on choisit
        # d'abord le document le plus proche du sens de la question
        best_doc = semantic_search(corpus_path, user_input)
        # Puis BERT extrait la reponse precise dans ce seul document
        answer = find_answer(user_input, best_doc)
        # None signifie qu'aucune reponse fiable n'a ete trouvee
        if answer is None:
            # Message d'excuse impose par le sujet
            print('A: Sorry, I do not understand your question.')
        else:
            # Affiche l'extrait trouve dans le meilleur document
            print('A: ' + answer)
