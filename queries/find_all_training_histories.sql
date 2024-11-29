CREATE PROCEDURE `FindAllTrainingHistories`(IN _train_models JSON, start INT, length INT)
BEGIN
	SELECT trainings.id, name, description, train_model, loss_function, class_weight, is_test_algorithm, execute_time, min_sample_count,
		(SELECT COUNT(1) FROM ddi_gnn.training_results WHERE trainings.id = training_results.training_id) as training_results_count,
		(SELECT COUNT(1) FROM ddi_gnn.training_result_details WHERE trainings.id = training_result_details.training_id) as training_result_details_count
	FROM trainings
	WHERE (_train_models is null or
		train_model IN (SELECT value FROM JSON_TABLE(_train_models, "$[*]" COLUMNS (value VARCHAR(10) PATH "$")) AS json_ids))
	GROUP BY trainings.id, name, description, train_model, is_test_algorithm, training_conditions, execute_time, min_sample_count, trainings.order
	ORDER BY trainings.order, trainings.id
    LIMIT length OFFSET start;
END