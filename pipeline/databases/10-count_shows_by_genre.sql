-- Nombre de séries liées à chaque genre, du genre le plus utilisé au moins utilisé
-- Le nom du genre vient de tv_genres, et COUNT compte les séries liées dans tv_show_genres ;
-- les alias genre et number_of_shows sont les noms de colonnes demandés
SELECT tv_genres.name AS genre, COUNT(tv_show_genres.show_id) AS number_of_shows
-- JOIN ne garde que les genres qui ont au moins une série, donc les genres vides n'apparaissent pas,
-- et ON relie les deux tables car genre_id pointe vers l'id du genre
FROM tv_genres JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
-- GROUP BY fait un groupe par genre pour que COUNT compte les séries de chaque genre séparément
GROUP BY genre
-- DESC met en premier les genres qui ont le plus de séries
ORDER BY number_of_shows DESC;
