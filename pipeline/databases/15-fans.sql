-- Ranks country origins of bands by their total number of fans
SELECT origin, SUM(metal_bands.fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
