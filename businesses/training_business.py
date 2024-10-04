import base64
import io
import json
import mimetypes
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from injector import inject
from tqdm import tqdm

from businesses.base_business import BaseBusiness
from businesses.trains import train_instances
from common.enums.category import Category
from common.enums.compare_plot_type import ComparePlotType
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType
from common.enums.train_models import TrainModel
from common.enums.training_result_type import TrainingResultType
from common.helpers import math_helper, embedding_helper, json_helper
from core.domain.training_result import TrainingResult
from core.domain.training_result_detail import TrainingResultDetail
from core.models.training_parameter_model import TrainingParameterModel
from core.view_models.train_request_view_model import TrainRequestViewModel
from core.repository_models.compare_plot_dto import ComparePlotDTO
from core.repository_models.interaction_dto import InteractionDTO
from core.repository_models.reduction_data_dto import ReductionDataDTO
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from core.repository_models.training_result_dto import TrainingResultDTO
from core.repository_models.training_scheduled_dto import TrainingScheduledDTO
from core.view_models.image_info_view_model import ImageInfoViewModel
from core.view_models.training_history_details_view_model import TrainingHistoryDetailsViewModel
from core.view_models.training_history_view_model import TrainingHistoryViewModel
from core.view_models.training_scheduled_view_model import TrainingScheduledViewModel
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository
from infrastructure.repositories.training_repository import TrainingRepository
from infrastructure.repositories.training_result_detail_repository import TrainingResultDetailRepository
from infrastructure.repositories.training_result_repository import TrainingResultRepository
from infrastructure.repositories.training_scheduled_repository import TrainingScheduledRepository


def map_training_data(interactions: list[InteractionDTO], reductions: list[ReductionDataDTO], category: Category) -> list[TrainingDataDTO]:
    reduction_dict = {reduction.drug_id: [float(val) for val in reduction.reduction_value[1:-1].split(',')]
                      for reduction in reductions}

    training_data = []

    for interaction in tqdm(interactions, 'Mapping training data'):
        training_entity = TrainingDataDTO(drug_1=interaction.drug_1,
                                          drugbank_id_1=interaction.drugbank_id_1,
                                          reduction_values_1=reduction_dict.get(interaction.drug_1, None),
                                          drug_2=interaction.drug_2,
                                          drugbank_id_2=interaction.drugbank_id_2,
                                          reduction_values_2=reduction_dict.get(interaction.drug_2, None),
                                          category=category,
                                          interaction_type=interaction.interaction_type)
        training_data.append(training_entity)

    return training_data


def map_training_history_view_model(results: list[TrainingResultDTO]) -> list[TrainingHistoryViewModel]:
    return [TrainingHistoryViewModel(
        id=item.id,
        name=item.name,
        description=item.description,
        train_model=item.train_model.name,
        loss_function=item.loss_function.display_name if item.loss_function else '',
        class_weight=item.class_weight,
        training_conditions=item.training_conditions,
        accuracy=item.accuracy,
        loss=item.loss,
        f1_score_weighted=item.f1_score_weighted,
        f1_score_micro=item.f1_score_micro,
        f1_score_macro=item.f1_score_macro,
        auc_weighted=item.auc_weighted,
        auc_micro=item.auc_micro,
        auc_macro=item.auc_macro,
        aupr_weighted=item.aupr_weighted,
        aupr_micro=item.aupr_micro,
        aupr_macro=item.aupr_macro,
        recall_weighted=item.recall_weighted,
        recall_micro=item.recall_micro,
        recall_macro=item.recall_macro,
        precision_weighted=item.precision_weighted,
        precision_micro=item.precision_micro,
        precision_macro=item.precision_macro,
        execute_time=item.execute_time
    ) for item in results]


def map_training_scheduled_view_model(results: list[TrainingScheduledDTO]) -> list[TrainingScheduledViewModel]:
    return [TrainingScheduledViewModel(
        id=item.id,
        name=item.name,
        description=item.description,
        train_model=item.train_model.name,
        training_conditions=item.training_conditions,
        schedule_date=item.schedule_date
    ) for item in results]


def map_training_history_details_view_model(results: list[TrainingResultDetailDTO]) \
        -> list[TrainingHistoryDetailsViewModel]:
    return [TrainingHistoryDetailsViewModel(
        training_label=item.training_label,
        accuracy=item.accuracy,
        f1_score=item.f1_score,
        auc=item.auc,
        aupr=item.aupr,
        recall=item.recall,
        precision=item.precision
    ) for item in results]


colors = list(mcolors.TABLEAU_COLORS.values())


class TrainingBusiness(BaseBusiness):
    @inject
    def __init__(self, reduction_repository: ReductionDataRepository,
                 training_repository: TrainingRepository,
                 training_result_repository: TrainingResultRepository,
                 training_result_detail_repository: TrainingResultDetailRepository,
                 training_scheduled_repository: TrainingScheduledRepository):
        BaseBusiness.__init__(self)
        self.reduction_repository = reduction_repository
        self.training_repository = training_repository
        self.training_result_repository = training_result_repository
        self.training_result_detail_repository = training_result_detail_repository
        self.training_scheduled_repository = training_scheduled_repository
        self.plot_folder_name = "training_plots"

    def schedule_train(self, train_request: TrainRequestViewModel):

        self.training_scheduled_repository.insert(train_request.name, train_request.description, train_request.train_model, train_request.loss_function,
                                                  train_request.class_weight, train_request.is_test_algorithm, train_request.to_json(),
                                                  datetime.now(timezone.utc))

    def run_trainings(self, training_id: int = None):

        if not training_id:
            training_schedules = self.training_scheduled_repository.find_overdue_training_scheduled()

            for training_schedule in training_schedules:
                try:
                    self.train(training_schedule)
                except Exception as ex:
                    print(f"An error occurred during training: {ex} for id: {training_schedule.id}")

        else:
            training_schedule = self.training_scheduled_repository.get_training_scheduled_by_id(training_id)

            self.train(training_schedule)

    def train(self, train_schedule: TrainingScheduledDTO):

        train_id = self.training_repository.insert(train_schedule.name, train_schedule.description,
                                                   train_schedule.train_model, train_schedule.loss_function, train_schedule.class_weight,
                                                   train_schedule.is_test_algorithm, train_schedule.training_conditions)

        data = self.prepare_data(TrainRequestViewModel.from_json(train_schedule.training_conditions), train_schedule.is_test_algorithm)

        print('Start training...')
        instance = train_instances.get_instance(train_schedule.train_model)
        training_result = instance.train(TrainingParameterModel(train_id=train_id,
                                                                loss_function=train_schedule.loss_function,
                                                                class_weight=train_schedule.class_weight,
                                                                is_test_algorithm=train_schedule.is_test_algorithm), data)

        print('Update...')
        self.training_repository.update(train_id, json.dumps(training_result.data_report, default=json_helper.convert_numpy_types))

        print('Training Results...')
        training_results = [TrainingResult(training_id=train_id, training_result_type=item.training_result_type, result_value=item.result_value)
                            for item in training_result.training_results]
        self.training_result_repository.insert_batch_check_duplicate(training_results)

        # model_binary = pickle.dumps(training_result.model)
        # model = pickle.loads(model_binary)
        # model.save(f'training_models/{train_id}.h5')

        with open(f'training_models/{train_id}.pkl', 'wb') as file:
            pickle.dump(training_result.model, file)

        train_result_details = [TrainingResultDetail(training_id=train_id,
                                                     training_label=item.training_label,
                                                     f1_score=item.f1_score,
                                                     accuracy=item.accuracy,
                                                     auc=item.auc,
                                                     aupr=item.aupr,
                                                     recall=item.recall,
                                                     precision=item.precision) for item in
                                training_result.training_result_details]

        self.training_result_detail_repository.insert_batch_check_duplicate(train_result_details)

        self.training_scheduled_repository.delete(train_schedule.id)

    def get_training_data(self, similarity_type: SimilarityType, category: Category,
                          reduction_category: ReductionCategory, interactions: list[InteractionDTO]) -> list[TrainingDataDTO]:

        reductions = self.reduction_repository.find_reductions(reduction_category, similarity_type, category,
                                                               True, True, True, True,
                                                               0, 100000)

        results = map_training_data(interactions, reductions, category)

        if not results:
            raise Exception(f"No training data found for {similarity_type} {category} {reduction_category}")

        return results

    def prepare_data(self, train_request: TrainRequestViewModel, is_test_algorithm: bool):

        data = []

        interactions = self.reduction_repository.find_interactions(True, True, True, True)

        if is_test_algorithm:
            interactions = self.stratified_sample(interactions, test_size=0.1, min_samples_per_category=5)

        if train_request.substructure_similarity and train_request.substructure_reduction:
            print('Fetching Substructure data!')
            data.append(self.get_training_data(train_request.substructure_similarity, Category.Substructure,
                                               train_request.substructure_reduction, interactions))

        if train_request.target_similarity and train_request.target_reduction:
            print('Fetching Target data!')
            data.append(self.get_training_data(train_request.target_similarity, Category.Target,
                                               train_request.target_reduction, interactions))

        if train_request.enzyme_similarity and train_request.enzyme_reduction:
            print('Fetching Enzyme data!')
            data.append(self.get_training_data(train_request.enzyme_similarity, Category.Enzyme,
                                               train_request.enzyme_reduction, interactions))

        if train_request.pathway_similarity and train_request.pathway_reduction:
            print('Fetching Pathway data!')
            data.append(self.get_training_data(train_request.pathway_similarity, Category.Pathway,
                                               train_request.pathway_reduction, interactions))

        if train_request.description_embedding and train_request.description_reduction:
            print('Fetching Description data!')
            category = embedding_helper.find_category(train_request.description_embedding, TextType.Description)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.description_reduction, interactions))

        if train_request.indication_embedding and train_request.indication_reduction:
            print('Fetching Indication data!')
            category = embedding_helper.find_category(train_request.indication_embedding, TextType.Indication)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.indication_reduction, interactions))

        if train_request.pharmacodynamics_embedding and train_request.pharmacodynamics_reduction:
            print('Fetching Pharmacodynamics data!')
            category = embedding_helper.find_category(train_request.pharmacodynamics_embedding,
                                                      TextType.Pharmacodynamics)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.pharmacodynamics_reduction, interactions))

        if train_request.mechanism_of_action_embedding and train_request.mechanism_of_action_reduction:
            print('Fetching Mechanism of action data!')
            category = embedding_helper.find_category(train_request.mechanism_of_action_embedding,
                                                      TextType.MechanismOfAction)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.mechanism_of_action_reduction, interactions))

        if train_request.toxicity_embedding and train_request.toxicity_reduction:
            print('Fetching Toxicity data!')
            category = embedding_helper.find_category(train_request.toxicity_embedding, TextType.Toxicity)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.toxicity_reduction, interactions))

        if train_request.metabolism_embedding and train_request.metabolism_reduction:
            print('Fetching Metabolism data!')
            category = embedding_helper.find_category(train_request.metabolism_embedding, TextType.Metabolism)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.metabolism_reduction, interactions))

        if train_request.absorption_embedding and train_request.absorption_reduction:
            print('Fetching Absorption data!')
            category = embedding_helper.find_category(train_request.absorption_embedding, TextType.Absorption)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.absorption_reduction, interactions))

        if train_request.half_life_embedding and train_request.half_life_reduction:
            print('Fetching Half life data!')
            category = embedding_helper.find_category(train_request.half_life_embedding, TextType.HalfLife)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.half_life_reduction, interactions))

        if train_request.protein_binding_embedding and train_request.protein_binding_reduction:
            print('Fetching Protein binding data!')
            category = embedding_helper.find_category(train_request.protein_binding_embedding,
                                                      TextType.ProteinBinding)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.protein_binding_reduction, interactions))

        if train_request.route_of_elimination_embedding and train_request.route_of_elimination_reduction:
            print('Fetching Route of elimination data!')
            category = embedding_helper.find_category(train_request.route_of_elimination_embedding,
                                                      TextType.RouteOfElimination)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.route_of_elimination_reduction, interactions))

        if train_request.volume_of_distribution_embedding and train_request.volume_of_distribution_reduction:
            print('Fetching Volume of distribution data!')
            category = embedding_helper.find_category(train_request.volume_of_distribution_embedding,
                                                      TextType.VolumeOfDistribution)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.volume_of_distribution_reduction, interactions))

        if train_request.clearance_embedding and train_request.clearance_reduction:
            print('Fetching Clearance data!')
            category = embedding_helper.find_category(train_request.clearance_embedding, TextType.Clearance)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.clearance_reduction, interactions))

        if train_request.classification_description_embedding and train_request.classification_description_reduction:
            print('Fetching Classification Description data!')
            category = embedding_helper.find_category(train_request.classification_description_embedding,
                                                      TextType.ClassificationDescription)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.classification_description_reduction, interactions))

        return data

    def get_training_scheduled(self, train_model: TrainModel):

        results = self.training_scheduled_repository.find_all_training_scheduled(train_model, 0, 100000)

        return map_training_scheduled_view_model(results)

    def get_history(self, train_model: TrainModel, start: int, length: int):

        total_number = self.training_repository.get_training_count(train_model)

        results = self.training_repository.find_all_training(train_model, start, length)

        return map_training_history_view_model(results), total_number[0]

    def get_history_details(self, train_id: int):

        results = self.training_result_detail_repository.find_all_training_result_details(train_id)

        return map_training_history_details_view_model(results)

    def get_history_conditions(self, train_id: int):

        result = self.training_repository.get_training_by_id(train_id)

        return result.training_conditions

    def get_history_data_reports(self, train_id: int):

        result = self.training_repository.get_training_by_id(train_id)

        return result.data_report

    def get_history_plots(self, train_id: int) -> list[ImageInfoViewModel]:

        image_folder = f'{self.plot_folder_name}/{train_id}'

        image_files: list[ImageInfoViewModel] = []

        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)

            if mimetypes.guess_type(file_path)[0] and 'image' in mimetypes.guess_type(file_path)[0]:
                file_name, _ = os.path.splitext(file)

                image_files.append(ImageInfoViewModel(path=f"{file_path}", name=file_name))

        return image_files

    def get_comparing_plot(self, train_result_ids: list[int], compare_plot_type: ComparePlotType) \
            -> ImageInfoViewModel:

        plot_info = self.get_plot_info(compare_plot_type, train_result_ids)

        generate_plt = self.generate_plot(plot_info)

        buf = io.BytesIO()
        generate_plt.savefig(buf, format='png')
        buf.seek(0)

        generate_plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')

        return ImageInfoViewModel(path=image_base64, name=plot_info.compare_plot_type.name)

    def generate_plot(self, plot: ComparePlotDTO):
        match plot.compare_plot_type.plot_name:
            case 'bar':
                return self.plot_bar(plot)
            case 'radial':
                return self.plot_radial(plot)

    @staticmethod
    def plot_bar(plot_data: ComparePlotDTO):
        plt.figure()

        x = range(len(plot_data.datas))

        bars = plt.bar(x, plot_data.datas, color=colors[:len(plot_data.datas)], label=plot_data.labels)

        for bar in bars:
            y_val = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, y_val, f'{y_val:.5f}', ha='center', va='bottom')

        plt.ylim(min(plot_data.datas) * 0.9, math_helper.round_up(max(plot_data.datas), 1))

        plt.xticks([])

        plt.title(plot_data.compare_plot_type.name)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        return plt

    @staticmethod
    def plot_radial(plot_data: ComparePlotDTO):
        num_categories = len(plot_data.datas[0])

        angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
        if angles:
            angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, data in enumerate(plot_data.datas):
            if len(data) > 0:
                data = np.concatenate((data, [data[0]]))
            else:
                continue

            ax.plot(angles, data, linewidth=2, linestyle='solid', label=plot_data.labels[i])

        category_labels = [i for i in range(num_categories)]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(category_labels, fontsize=8)

        valid_datas = [d for d in plot_data.datas if len(d) > 0]

        if valid_datas:
            ax.set_ylim(min([min(d) for d in valid_datas]) * 0.9,
                        math_helper.round_up(max([max(d) for d in valid_datas]), 1))
        else:
            # Handle case where all datas are empty, e.g., set a default y-limit
            ax.set_ylim(0, 1)

        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        plt.title(plot_data.compare_plot_type.name)

        return plt

    def get_plot_info(self, compare_plot_type: ComparePlotType, train_ids: list[int]) -> ComparePlotDTO:
        match compare_plot_type:
            case ComparePlotType.Accuracy:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.accuracy)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Loss:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.loss)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.F1_Score_Weighted:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.f1_score_weighted)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.F1_Score_Micro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.f1_score_micro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.F1_Score_Macro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.f1_score_macro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUC_Weighted:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.auc_weighted)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUC_Micro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.auc_micro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUC_Macro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.auc_macro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUPR_Weighted:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.aupr_weighted)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUPR_Micro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.aupr_micro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUPR_Macro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.aupr_macro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Recall_Weighted:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.recall_weighted)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Recall_Micro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.recall_micro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Recall_Macro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.recall_macro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Precision_Weighted:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.precision_weighted)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Precision_Micro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.precision_micro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Precision_Macro:
                data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.precision_macro)
                               for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.result_value for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Details_Accuracy:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_id)
                    for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.accuracy for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details if len(r) > 0])

            case ComparePlotType.Details_F1_Score:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_id)
                    for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.f1_score for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details if len(r) > 0])

            case ComparePlotType.Details_AUC:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_id)
                    for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.auc for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details if len(r) > 0])

            case ComparePlotType.Details_AUPR:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_id)
                    for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.aupr for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details if len(r) > 0])

            case ComparePlotType.Details_Recall:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_id)
                    for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.recall for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details if len(r) > 0])

            case ComparePlotType.Details_Precision:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_id)
                    for train_id in train_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.precision for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details if len(r) > 0])

    @staticmethod
    def group_by_category(data):
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item.interaction_type].append(item)
        return grouped_data

    def stratified_sample(self, data: list[InteractionDTO], test_size=0.1, min_samples_per_category=5) -> list[InteractionDTO]:
        grouped_data = self.group_by_category(data)

        test_data = []

        for category, items in grouped_data.items():
            sample_size = max(int(len(items) * test_size), min_samples_per_category)

            if len(items) <= sample_size:
                test_data.extend(items)
            else:
                test_data.extend(random.sample(items, sample_size))

        return test_data
