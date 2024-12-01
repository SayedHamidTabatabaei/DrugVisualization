CREATE DEFINER=`admin`@`localhost` PROCEDURE `FindAllTrainings`(IN _train_models JSON, _create_date datetime, _min_sample_count int, start INT, length INT)
BEGIN
	SELECT trainings.id, name, description, train_model, loss_function, class_weight, is_test_algorithm, training_conditions,
		model_parameters, data_report, execute_time, min_sample_count,
		sum(case training_result_type when 1 then result_value else 0 end) as accuracy,
		sum(case training_result_type when 2 then result_value else 0 end) as loss,
		sum(case training_result_type when 3 then result_value else 0 end) as f1_score_weighted,
		sum(case training_result_type when 4 then result_value else 0 end) as f1_score_micro,
		sum(case training_result_type when 5 then result_value else 0 end) as f1_score_macro,
		sum(case training_result_type when 6 then result_value else 0 end) as auc_weighted,
		sum(case training_result_type when 7 then result_value else 0 end) as auc_micro,
		sum(case training_result_type when 8 then result_value else 0 end) as auc_macro,
		sum(case training_result_type when 9 then result_value else 0 end) as aupr_weighted,
		sum(case training_result_type when 10 then result_value else 0 end) as aupr_micro,
		sum(case training_result_type when 11 then result_value else 0 end) as aupr_macro,
		sum(case training_result_type when 12 then result_value else 0 end) as recall_weighted,
		sum(case training_result_type when 13 then result_value else 0 end) as recall_micro,
		sum(case training_result_type when 14 then result_value else 0 end) as recall_macro,
		sum(case training_result_type when 15 then result_value else 0 end) as precision_weighted,
		sum(case training_result_type when 16 then result_value else 0 end) as precision_micro,
		sum(case training_result_type when 17 then result_value else 0 end) as precision_macro
	FROM trainings INNER JOIN training_results ON trainings.id = training_results.training_id
    WHERE (_train_models is null or
		train_model IN (SELECT value FROM JSON_TABLE(_train_models, "$[*]" COLUMNS (value VARCHAR(10) PATH "$")) AS json_ids)) and
        (_create_date IS NULL OR execute_time > _create_date) and (_min_sample_count is null or min_sample_count = _min_sample_count) and
        trainings.order <> 0
	GROUP BY trainings.id, name, description, train_model, is_test_algorithm, training_conditions, execute_time, min_sample_count, trainings.order
    ORDER BY trainings.order, trainings.id
    LIMIT length OFFSET start;
END