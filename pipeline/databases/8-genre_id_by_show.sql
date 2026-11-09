-- Liste les séries qui ont au moins un genre, avec l'id de chacun de leurs genres,
-- triées par titre puis par genre_id
-- Le titre vient de tv_shows et le genre_id de tv_show_genres : l'info est répartie sur 2 tables
SELECT tv_shows.title, tv_show_genres.genre_id
-- JOIN ne garde que les séries qui ont une ligne dans tv_show_genres (donc au moins un genre),
-- et ON relie les deux tables car show_id pointe vers l'id de la série
FROM tv_shows JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
-- Trier aussi par genre_id départage les lignes d'une même série qui a plusieurs genres
ORDER BY tv_shows.title, tv_show_genres.genre_id ASC;
