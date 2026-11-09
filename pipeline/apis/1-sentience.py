#!/usr/bin/env python3
"""Module that queries the SWAPI to list home planets of sentient species."""
import requests


def sentientPlanets():
    """
    Return the list of names of the home planets of all sentient species.

    A species is considered sentient if the word "sentient" appears in
    either its designation or its classification attribute.

    Returns:
        list: names of the home planets of sentient species. Species
            without a known homeworld are skipped.
    """
    # Point de départ : la première page de la liste des espèces
    url = "https://swapi-api.hbtn.io/api/species/"
    # On accumule les noms ici car les résultats arrivent page par page
    planets = []
    # L'API est paginée : on boucle tant qu'il reste une page suivante
    while url:
        # .json() transforme la réponse HTTP en dict Python exploitable
        data = requests.get(url).json()
        # "results" contient seulement les espèces de la page courante
        for species in data["results"]:
            # "sentient" peut apparaître dans l'un OU l'autre des deux
            # champs selon l'espèce, donc on vérifie les deux
            if ("sentient" in species["designation"] or
                    "sentient" in species["classification"]):
                # L'espèce ne donne qu'une URL vers sa planète, pas son nom
                planet_url = species["homeworld"]
                # Certaines espèces n'ont pas de planète (None) : on les
                # saute sinon requests.get(None) planterait
                if planet_url:
                    # 2e requête nécessaire pour récupérer le nom réel
                    planet = requests.get(planet_url).json()
                    # On garde uniquement le nom, c'est ce qui est demandé
                    planets.append(planet["name"])
        # "next" vaut None à la dernière page, ce qui arrête la boucle
        url = data["next"]
    # Liste des planètes d'origine des espèces sentientes
    return planets
