#!/usr/bin/env python3
"""Module that queries the SWAPI to list starships by passenger capacity."""
import requests


def availableShips(passengerCount):
    """
    Return the list of ships that can hold a given number of passengers.

    Args:
        passengerCount (int): minimum number of passengers the ship
            must be able to carry.

    Returns:
        list: names of the starships whose passenger capacity is greater
            than or equal to passengerCount. Empty list if none match.
    """
    # Point de départ : la première page de la liste des vaisseaux
    url = "https://swapi-api.hbtn.io/api/starships/"
    # On accumule les noms ici car les résultats arrivent page par page
    ships = []
    # L'API est paginée : on boucle tant qu'il reste une page suivante
    while url:
        # .json() transforme la réponse HTTP en dict Python exploitable
        data = requests.get(url).json()
        # "results" contient seulement les vaisseaux de la page courante
        for ship in data["results"]:
            # On enlève les virgules ("1,200" -> "1200") sinon int() plante
            place = ship["passengers"].replace(",", "")
            # isdigit() écarte "n/a" / "unknown" avant la conversion en int
            if place.isdigit() and int(place) >= passengerCount:
                # Le vaisseau a assez de places : on garde son nom
                ships.append(ship["name"])
        # "next" vaut None à la dernière page, ce qui arrête la boucle
        url = data["next"]
    # Liste vide si aucun vaisseau ne convient, comme demandé
    return ships
