-- Température moyenne par ville, de la plus chaude à la plus froide ;
-- GROUP BY city fait une moyenne par ville au lieu d'une moyenne globale,
-- ORDER BY avg_temp DESC trie sur cette moyenne en partant du plus chaud
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
