-- Liste score puis nom des lignes avec un score >= 10 ; WHERE écarte les
-- petits scores, ORDER BY ... DESC met le meilleur score en premier
SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC;
