#!/usr/bin/env python3
"""
Script that prints the location of a specific GitHub user.

The full API URL of the user is passed as the first command-line
argument. If the user does not exist, it prints "Not found". If the
rate limit is exceeded, it prints "Reset in X min" where X is the
number of minutes before the rate limit resets.
"""
import sys
import time
import requests

# Le code ne s'exécute que si le fichier est lancé directement,
# pas quand il est importé par un autre module
if __name__ == '__main__':
    # L'URL complète de l'utilisateur est passée en argument du script
    url = sys.argv[1]
    # On garde la réponse entière (pas juste .json()) car on a besoin
    # du status_code et des headers pour gérer les différents cas
    r = requests.get(url)

    # 200 = l'utilisateur existe, la réponse contient ses infos
    if r.status_code == 200:
        # On n'affiche que le champ "location", c'est ce qui est demandé
        print(r.json()["location"])
    # 404 = l'utilisateur n'existe pas sur GitHub
    elif r.status_code == 404:
        # Message imposé par la consigne dans ce cas
        print("Not found")
    # 403 = GitHub bloque car on a dépassé la limite de requêtes
    elif r.status_code == 403:
        # Le header donne l'heure du reset en timestamp Unix (secondes),
        # converti en int car les headers sont toujours des chaînes
        reset = int(r.headers["X-Ratelimit-Reset"])
        # Timestamp actuel, dans la même unité, pour faire la différence
        now = time.time()
        # Différence en secondes / 60 pour l'avoir en minutes,
        # int() tronque pour afficher un nombre entier de minutes
        minutes = int((reset - now) / 60)
        # Message imposé par la consigne avec le temps d'attente restant
        print("Reset in {} min".format(minutes))
