import time
from typing import Union

from common.enums.category import Category
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from core.domain.reduction_data import ReductionData
from core.mappers import reduction_data_mapper
from core.repository_models.interaction_dto import InteractionDTO
from core.repository_models.reduction_data_dto import ReductionDataDTO
from core.repository_models.training_data_dto import TrainingDataDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class ReductionDataRepository(MySqlRepository):
    def __init__(self):
        super().__init__('reduction_data')

    def insert(self, drug_id: int, similarity_type: SimilarityType, category: Category,
               reduction_category: ReductionCategory, reduction_values: list[float],
               has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool) \
            -> ReductionData:
        reduction_data = ReductionData(drug_id=drug_id,
                                       similarity_type=similarity_type,
                                       category=category,
                                       reduction_category=reduction_category,
                                       reduction_values=str(reduction_values),
                                       has_enzyme=has_enzyme,
                                       has_pathway=has_pathway,
                                       has_target=has_target,
                                       has_smiles=has_smiles)

        super().insert(reduction_data)

        return reduction_data

    def insert_if_not_exits(self, drug_id: int, similarity_type: SimilarityType, category: Category,
                            reduction_category: ReductionCategory, reduction_values: list[float],
                            has_enzyme: bool, has_pathway: bool, has_target: bool,
                            has_smiles: bool) -> Union[ReductionData, None]:
        is_exists = self.is_exists_reduction_data(drug_id, similarity_type, category, reduction_category,
                                                  has_enzyme, has_pathway, has_target, has_smiles)

        if is_exists:
            return None

        reduction_data = self.insert(drug_id=drug_id,
                                     similarity_type=similarity_type,
                                     category=category,
                                     reduction_category=reduction_category,
                                     reduction_values=reduction_values,
                                     has_enzyme=has_enzyme,
                                     has_pathway=has_pathway,
                                     has_target=has_target,
                                     has_smiles=has_smiles)

        return reduction_data

    def insert_batch_check_duplicate(self, reduction_datas: list[ReductionData]):
        super().insert_batch_check_duplicate(reduction_datas, [ReductionData.reduction_values])

    def is_exists_reduction_data(self, drug_id: int, similarity_type: SimilarityType, category: Category,
                                 reduction_category: ReductionCategory,
                                 has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool) \
            -> bool:
        result, _ = self.call_procedure('FindReductionData',
                                        [drug_id, similarity_type.value, category.value, reduction_category,
                                         has_enzyme, has_pathway, has_target, has_smiles])

        reduction_data = result[0]

        return (reduction_data is not None and
                (reduction_data != [] if isinstance(reduction_data, list) else bool(reduction_data)))

    def find_training_data(self, similarity_type: SimilarityType,
                           category: Category,
                           reduction_category: ReductionCategory,
                           has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool) \
            -> list[TrainingDataDTO]:
        result, _ = self.call_procedure('FindTrainingData',
                                        [similarity_type.value, category.value, reduction_category.value,
                                         has_enzyme, has_pathway, has_target, has_smiles])

        start_time = time.time()
        print(f"Task started in {start_time}")

        map_result = reduction_data_mapper.map_training_data(result[0])

        end_time = time.time()

        duration = end_time - start_time
        print(f"Task completed in {duration} seconds. Started in {start_time} and Ended in {end_time}")

        return map_result

    def find_interactions(self, has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool) \
            -> list[InteractionDTO]:
        result, _ = self.call_procedure('FindInteractions',
                                        [has_enzyme, has_pathway, has_target, has_smiles])

        return reduction_data_mapper.map_interactions(result[0])

    def find_reductions(self, reduction_category: ReductionCategory,
                        similarity_type: SimilarityType,
                        category: Category,
                        has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool,
                        start: int, length: int) \
            -> list[ReductionDataDTO]:
        result, _ = self.call_procedure('FindReductionData',
                                        [similarity_type.value, category.value, reduction_category.value,
                                         has_enzyme, has_pathway, has_target, has_smiles, start, length])

        return reduction_data_mapper.map_reduction_data(result[0])

    def get_reduction_count(self, reduction_category: ReductionCategory,
                            similarity_type: SimilarityType,
                            category: Category,
                            has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool):
        result, _ = self.call_procedure('GetReductionCount',
                                        [similarity_type.value, category.value, reduction_category.value,
                                         has_enzyme, has_pathway, has_target, has_smiles])

        return result[0]
