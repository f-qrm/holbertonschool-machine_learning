-- View listing students with a score under 80 and no meeting in the last month
CREATE VIEW need_meeting AS
SELECT name FROM students
WHERE score < 80 AND (last_meeting IS NULL OR last_meeting < DATE_SUB(CURDATE(), INTERVAL 1 MONTH));
