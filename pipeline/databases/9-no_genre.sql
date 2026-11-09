-- Liste les séries qui n'ont aucun genre, triées par titre puis par genre_id
-- Le titre vient de tv_shows et le genre_id de tv_show_genres (il vaudra NULL ici)
SELECT tv_shows.title, tv_show_genres.genre_id
-- LEFT JOIN garde toutes les séries, même sans ligne dans tv_show_genres :
-- dans ce cas les colonnes de tv_show_genres sont remplies avec NULL
FROM tv_shows LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
-- Un genre_id NULL veut dire qu'aucun genre n'a été trouvé, donc on ne garde que ces séries
-- (on utilise IS NULL car "= NULL" ne vaut jamais vrai en SQL)
WHERE tv_show_genres.genre_id IS NULL
-- Le tri par genre_id est demandé par la consigne, même s'il vaut toujours NULL ici
ORDER BY tv_shows.title, tv_show_genres.genre_id ASC;
