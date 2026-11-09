-- Crée la table first_table ; IF NOT EXISTS pour que le script ne plante pas
-- si la table a déjà été créée lors d'un essai précédent
CREATE TABLE IF NOT EXISTS first_table(
    -- INT car l'id est un identifiant numérique entier
    id INT,
    -- VARCHAR(256) : texte de longueur variable, MySQL ne stocke que les
    -- caractères réellement utilisés (jusqu'à 256) au lieu d'une taille fixe
    name VARCHAR(256)
);
