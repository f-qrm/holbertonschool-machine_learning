#!/usr/bin/env python3
"""Interactive prompt loop that exits on a goodbye word."""

# Variantes courantes pour signaler qu'on veut arreter la boucle,
# comparees en minuscules pour ne pas dependre de la casse saisie
exit_words = ('exit', 'quit', 'goodbye', 'bye')
# Boucle infinie : on ne sait pas a l'avance combien de questions
# l'utilisateur va poser, seul un mot de sortie l'arrete
while True:
    # input() bloque jusqu'a ce que l'utilisateur tape une ligne
    user_input = input('Q: ')
    # lower() pour que "EXIT" ou "Bye" soient aussi reconnus
    if user_input.lower() in exit_words:
        # Message de fin demande par le sujet avant de quitter
        print("A: Goodbye")
        # break sort de la boucle infinie, et donc du programme
        break
    # Pas encore de modele ici : on affiche juste une reponse vide
    print("A:")
