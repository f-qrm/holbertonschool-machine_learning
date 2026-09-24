#!/usr/bin/env python3
"""Answer questions from a reference text in an interactive loop."""
# Import dynamique car le nom du fichier commence par un chiffre,
# ce qui interdit un "import" classique
question_answer = __import__('0-qa').question_answer

# Variantes courantes pour signaler qu'on veut arreter la boucle,
# comparees en minuscules pour ne pas dependre de la casse saisie
exit_words = ('exit', 'quit', 'goodbye', 'bye')


def answer_loop(reference):
    """
    Answer questions from a reference text until the user exits.

    Args:
        reference (str): the reference text in which to look for the
            answers.

    Returns:
        None
    """
    # Boucle infinie : seul un mot de sortie arrete la conversation
    while True:
        # input() bloque jusqu'a ce que l'utilisateur tape une ligne
        user_input = input('Q: ')
        # Verifie la commande de sortie avant d'appeler le modele,
        # pour ne pas gaspiller un appel BERT inutile sur "exit"
        if user_input.lower() in exit_words:
            # Message de fin demande par le sujet
            print("A: Goodbye")
            # Sort de la boucle, ce qui termine la fonction
            break
        # Reutilise la fonction du task 0 : meme modele, meme logique
        # d'extraction de reponse dans reference
        answer = question_answer(user_input, reference)
        # question_answer renvoie None quand aucune reponse fiable
        # n'a ete trouvee (confiance trop basse ou span incoherent)
        if answer is None:
            # Message d'excuse impose par le sujet
            print('A: Sorry, I do not understand your question.')
        else:
            # Affiche l'extrait trouve dans reference
            print('A: ' + answer)
