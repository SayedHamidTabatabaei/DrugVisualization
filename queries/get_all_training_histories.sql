CREATE DEFINER=`admin`@`localhost` PROCEDURE `GetAllTrainingCount`(IN _train_models JSON)
BEGIN
	SELECT COUNT(1) FROM trainings
    WHERE (_train_models is null or
		train_model IN (SELECT value FROM JSON_TABLE(_train_models, "$[*]" COLUMNS (value VARCHAR(10) PATH "$")) AS json_ids));
END