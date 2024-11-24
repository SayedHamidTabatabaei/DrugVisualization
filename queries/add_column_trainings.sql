ALTER TABLE `ddi_gnn`.`trainings`
ADD COLUMN `order` INT NOT NULL DEFAULT 100000 AFTER `min_sample_count`;
