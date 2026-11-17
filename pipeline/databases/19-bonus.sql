-- Procédure qui ajoute une correction, en créant le projet s'il n'existe pas
DELIMITER $$
CREATE PROCEDURE AddBonus(IN user_id INT, IN project_name VARCHAR(255), IN score INT)
BEGIN
    DECLARE p_id INT;

    SELECT id INTO p_id FROM projects WHERE name = project_name LIMIT 1;

    IF p_id IS NULL THEN
        INSERT INTO projects (name) VALUES (project_name);
        SET p_id = LAST_INSERT_ID();
    END IF;

    INSERT INTO corrections (user_id, project_id, score)
    VALUES (user_id, p_id, score);
END$$
DELIMITER ;
