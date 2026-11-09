-- Température maximale de chaque état, triée par nom d'état ;
-- GROUP BY state fait un groupe par état pour que MAX donne le max de chacun,
-- ORDER BY state range les états par ordre alphabétique
SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state;
