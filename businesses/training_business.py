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
from common.enums.embedding_type import EmbeddingType
from common.enums.scenarios import Scenarios
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType
from common.enums.train_models import TrainModel
from common.enums.training_result_type import TrainingResultType
from common.helpers import math_helper, embedding_helper, json_helper, smiles_helper
from core.domain.training_result import TrainingResult
from core.domain.training_result_detail import TrainingResultDetail
from core.models.training_parameter_base_model import TrainingParameterBaseModel
from core.models.training_parameter_models.fold_interaction_training_parameter_model import FoldInteractionTrainingParameterModel
from core.models.training_parameter_models.split_drugs_test_with_test_training_parameter_model import SplitDrugsTestWithTestTrainingParameterModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.compare_plot_dto import ComparePlotDTO
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_drug_train_values_dto import TrainingDrugTrainValuesDTO
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from core.repository_models.training_result_dto import TrainingResultDTO
from core.repository_models.training_scheduled_dto import TrainingScheduledDTO
from core.view_models.image_info_view_model import ImageInfoViewModel
from core.view_models.train_request_view_model import TrainRequestViewModel
from core.view_models.training_history_details_view_model import TrainingHistoryDetailsViewModel
from core.view_models.training_history_view_model import TrainingHistoryViewModel
from core.view_models.training_scheduled_view_model import TrainingScheduledViewModel
from infrastructure.repositories.drug_embedding_repository import DrugEmbeddingRepository
from infrastructure.repositories.drug_interaction_repository import DrugInteractionRepository
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.enzyme_repository import EnzymeRepository
from infrastructure.repositories.pathway_repository import PathwayRepository
from infrastructure.repositories.similarity_repository import SimilarityRepository
from infrastructure.repositories.target_repository import TargetRepository
from infrastructure.repositories.training_repository import TrainingRepository
from infrastructure.repositories.training_result_detail_repository import TrainingResultDetailRepository
from infrastructure.repositories.training_result_repository import TrainingResultRepository
from infrastructure.repositories.training_scheduled_repository import TrainingScheduledRepository


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
        execute_time=item.execute_time,
        min_sample_count=item.min_sample_count
    ) for item in results]


def map_training_scheduled_view_model(results: list[TrainingScheduledDTO]) -> list[TrainingScheduledViewModel]:
    return [TrainingScheduledViewModel(
        id=item.id,
        name=item.name,
        description=item.description,
        train_model=item.train_model.name,
        training_conditions=item.training_conditions,
        schedule_date=item.schedule_date,
        min_sample_count=item.min_sample_count
    ) for item in results]


def map_training_history_details_view_model(results: list[TrainingResultDetailDTO]) -> list[TrainingHistoryDetailsViewModel]:
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
    def __init__(self,
                 target_repository: TargetRepository,
                 enzyme_repository: EnzymeRepository,
                 pathway_repository: PathwayRepository,
                 drug_repository: DrugRepository,
                 training_repository: TrainingRepository,
                 training_result_repository: TrainingResultRepository,
                 training_result_detail_repository: TrainingResultDetailRepository,
                 training_scheduled_repository: TrainingScheduledRepository,
                 similarity_repository: SimilarityRepository,
                 drug_embedding_repository: DrugEmbeddingRepository,
                 drug_interaction_repository: DrugInteractionRepository):
        BaseBusiness.__init__(self)
        self.target_repository = target_repository
        self.enzyme_repository = enzyme_repository
        self.pathway_repository = pathway_repository
        self.drug_repository = drug_repository
        self.training_repository = training_repository
        self.training_result_repository = training_result_repository
        self.training_result_detail_repository = training_result_detail_repository
        self.training_scheduled_repository = training_scheduled_repository
        self.similarity_repository = similarity_repository
        self.drug_embedding_repository = drug_embedding_repository
        self.drug_interaction_repository = drug_interaction_repository
        self.plot_folder_name = "training_plots"

    def schedule_train(self, train_request: TrainRequestViewModel):

        self.training_scheduled_repository.insert(train_request.name, train_request.description, train_request.train_model, train_request.loss_function,
                                                  train_request.class_weight, train_request.is_test_algorithm, train_request.to_json(),
                                                  datetime.now(timezone.utc), train_request.min_sample_count)

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
                                                   train_schedule.is_test_algorithm, train_schedule.min_sample_count, train_schedule.training_conditions)

        print('Start training...')
        instance = train_instances.get_instance(train_schedule.train_model)

        training_parameter_model = self.build_training_parameter_model(train_id, train_schedule)
        training_result = instance.train(training_parameter_model)

        print('Update...')
        self.training_repository.update(train_id,
                                        data_report=json.dumps(training_result.data_report, default=json_helper.convert_numpy_types),
                                        model_parameters=json.dumps(training_result.model_info, default=json_helper.convert_numpy_types),
                                        fold_result_details=json.dumps(training_result.fold_result_details, default=json_helper.convert_numpy_types))

        print('Training Results...')
        training_results = [TrainingResult(training_id=train_id, training_result_type=item.training_result_type, result_value=item.result_value)
                            for item in training_result.training_results]
        self.training_result_repository.insert_batch_check_duplicate(training_results)

        if isinstance(training_result.model, bytes):
            with open(f'training_models/{train_id}.pkl', 'wb') as file:
                pickle.dump(training_result.model, file)
        elif isinstance(training_result.model, list):
            for idx in range(len(training_result.model)):
                with open(f'training_models/{train_id}_{idx}.pkl', 'wb') as file:
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

    def build_training_parameter_model(self, train_id, train_schedule) -> TrainingParameterBaseModel:

        train_request = TrainRequestViewModel.from_json(train_schedule.training_conditions)

        drug_data = self.prepare_drug_data(train_request)

        interaction_data = self.prepare_interaction_data(train_request, train_schedule.min_sample_count)

        if train_schedule.train_model.scenario == Scenarios.SplitInteractionSimilarities:

            # if train_schedule.is_test_algorithm:
            #     interaction_data = self.stratified_sample(interaction_data, test_size=0.01, min_samples_per_category=5)

            return SplitInteractionSimilaritiesTrainingParameterModel(train_id=train_id,
                                                                      loss_function=train_schedule.loss_function,
                                                                      class_weight=train_schedule.class_weight,
                                                                      is_test_algorithm=train_schedule.is_test_algorithm,
                                                                      drug_data=drug_data,
                                                                      interaction_data=interaction_data)

        elif train_schedule.train_model.scenario == Scenarios.SplitDrugsTestWithTrain:

            return SplitDrugsTestWithTrainTrainingParameterModel(train_id=train_id,
                                                                 loss_function=train_schedule.loss_function,
                                                                 class_weight=train_schedule.class_weight,
                                                                 is_test_algorithm=train_schedule.is_test_algorithm,
                                                                 drug_data=drug_data,
                                                                 interaction_data=interaction_data)
        elif train_schedule.train_model.scenario == Scenarios.SplitDrugsTestWithTest:

            return SplitDrugsTestWithTestTrainingParameterModel(train_id=train_id,
                                                                loss_function=train_schedule.loss_function,
                                                                class_weight=train_schedule.class_weight,
                                                                is_test_algorithm=train_schedule.is_test_algorithm,
                                                                drug_data=drug_data,
                                                                interaction_data=interaction_data)
        elif train_schedule.train_model.scenario == Scenarios.SplitDrugsTestWithTest:

            return SplitDrugsTestWithTestTrainingParameterModel(train_id=train_id,
                                                                loss_function=train_schedule.loss_function,
                                                                class_weight=train_schedule.class_weight,
                                                                is_test_algorithm=train_schedule.is_test_algorithm,
                                                                drug_data=drug_data,
                                                                interaction_data=interaction_data)
        elif train_schedule.train_model.scenario == Scenarios.FoldInteractionSimilarities:

            return FoldInteractionTrainingParameterModel(train_id=train_id,
                                                         loss_function=train_schedule.loss_function,
                                                         class_weight=train_schedule.class_weight,
                                                         is_test_algorithm=train_schedule.is_test_algorithm,
                                                         drug_data=drug_data,
                                                         interaction_data=interaction_data)

    def get_training_scheduled(self, train_model: TrainModel):

        results = self.training_scheduled_repository.find_all_training_scheduled(train_model, 0, 100000)

        return map_training_scheduled_view_model(results)

    def get_history(self, scenario: Scenarios, train_model: TrainModel, create_date: datetime, min_sample_count: int, start: int, length: int):
        if train_model:
            train_models = [train_model]
        elif scenario:
            train_models = [train_model for train_model in TrainModel if train_model.scenario == scenario]
        else:
            train_models = None

        total_number = self.training_repository.get_training_count(train_models, create_date=create_date, min_sample_count=min_sample_count)

        results = self.training_repository.find_all_training(train_models, create_date=create_date, min_sample_count=min_sample_count,
                                                             start=start, length=length)

        return map_training_history_view_model(results), total_number[0]

    def get_history_details(self, train_id: int):

        results = self.training_result_detail_repository.find_all_training_result_details(train_id)

        return map_training_history_details_view_model(results)

    def get_history_conditions(self, train_id: int):

        result = self.training_repository.get_training_by_id(train_id)

        return result.training_conditions

    def get_history_model_information(self, train_id: int):

        result = self.training_repository.get_training_by_id(train_id)

        return result.model_parameters

    def get_history_data_reports(self, train_id: int):

        result = self.training_repository.get_training_by_id(train_id)

        return result.data_report

    def get_history_fold_result_details(self, train_id: int):

        result = self.training_repository.get_training_by_id(train_id)

        return result.fold_result_details

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

    def get_per_category_result_details(self, train_result_ids: list[int], compare_plot_type: ComparePlotType):

        result_details = self.training_result_detail_repository.find_training_result_details_by_training_ids(train_result_ids)

        column_names = list({f'{detail.training_name}({detail.training_id})' for detail in result_details})
        column_names = ['level'] + column_names

        scores = defaultdict(dict)

        for detail in result_details:
            if compare_plot_type == ComparePlotType.Details_Accuracy:
                scores[detail.training_label][detail.training_id] = detail.accuracy
            elif compare_plot_type == ComparePlotType.Details_F1_Score:
                scores[detail.training_label][detail.training_id] = detail.f1_score
            elif compare_plot_type == ComparePlotType.Details_AUC:
                scores[detail.training_label][detail.training_id] = detail.auc
            elif compare_plot_type == ComparePlotType.Details_AUPR:
                scores[detail.training_label][detail.training_id] = detail.aupr
            elif compare_plot_type == ComparePlotType.Details_Recall:
                scores[detail.training_label][detail.training_id] = detail.recall
            elif compare_plot_type == ComparePlotType.Details_Precision:
                scores[detail.training_label][detail.training_id] = detail.precision

        unique_training_ids = sorted({detail.training_id for detail in result_details})

        result = [
            (label,) + tuple(scores[label].get(training_id, None) for training_id in unique_training_ids)
            for label in sorted(scores.keys())
        ]

        data = [dict(zip(column_names, row)) for row in result]

        return column_names, data

    def get_data_report(self, train_result_ids: list[int], data_set_name: str):
        results = self.training_repository.get_training_by_ids(train_result_ids)

        combined_train_interaction_ids = []

        columns = [f"{r.name}({r.id})" for r in results]

        for result in results:
            with open(f'seeds/{result.id}.json', 'r') as file:
                data = json.load(file)

                combined_train_interaction_ids.append(data.get(data_set_name, []))

        return columns, self.full_join_multiple_lists(columns, combined_train_interaction_ids)

    def get_data_summary(self, train_result_ids: list[int]):
        results = self.training_repository.get_training_by_ids(train_result_ids)

        training_names = list({f'{result.name}({result.id})' for result in results})

        column_names = [f'Total {name}' for name in training_names] + [f'Train {name}' for name in training_names] + [f'Test {name}' for name in training_names]

        column_names = ['level'] + column_names

        data_dict = {'level': []}

        for result in results:
            # Create column names
            total_col_name = f'Total {result.name}({result.id})'
            train_col_name = f'Train {result.name}({result.id})'
            test_col_name = f'Test {result.name}({result.id})'

            data_report = json.loads(result.data_report)

            total_report = {list(d.keys())[0]: list(d.values())[0] for d in data_report['total_report']}
            train_report = {list(d.keys())[0]: list(d.values())[0] for d in data_report['train_report']}
            test_report = {list(d.keys())[0]: list(d.values())[0] for d in data_report['test_report']}

            if not data_dict['level']:
                data_dict['level'] = ['Total'] + list(total_report.keys())

            data_dict[total_col_name] = [data_report['total_count']] + list(total_report.values())
            data_dict[train_col_name] = [data_report['train_count']] + list(train_report.values())
            data_dict[test_col_name] = [data_report['test_count']] + list(test_report.values())

        return column_names, self.convert_data_to_structured_format(data_dict)

    @staticmethod
    def full_join_multiple_lists(trains, lists):

        full_set = sorted(set().union(*lists))
        sets = [set(lst) for lst in lists]

        result = []
        for item in full_set:
            item_dict = {trains[i]: item if item in sets[i] else None for i in range(len(sets))}
            result.append(item_dict)

        return result

    @staticmethod
    def convert_data_to_structured_format(data):

        levels = data['level']
        reordered_levels = ['Total'] + sorted([level for level in levels if level != 'Total'], key=int)

        rows = []

        for level in reordered_levels:

            row = {'level': level}

            for key in data.keys():
                if key != 'level':
                    row[key] = data[key][reordered_levels.index(level)]

            rows.append(row)

        return rows

    def generate_plot(self, plot: ComparePlotDTO):
        if plot.compare_plot_type.plot_name == 'bar':
            return self.plot_bar(plot)
        elif plot.compare_plot_type.plot_name == 'radial':
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

        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='x-small')

        plt.title(plot_data.compare_plot_type.name)

        return plt

    def get_plot_info(self, compare_plot_type: ComparePlotType, train_ids: list[int]) -> ComparePlotDTO:
        if compare_plot_type == ComparePlotType.Accuracy:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.accuracy)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Loss:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.loss)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.F1_Score_Weighted:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.f1_score_weighted)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.F1_Score_Micro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.f1_score_micro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.F1_Score_Macro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.f1_score_macro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.AUC_Weighted:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.auc_weighted)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.AUC_Micro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.auc_micro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.AUC_Macro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.auc_macro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.AUPR_Weighted:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.aupr_weighted)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.AUPR_Micro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.aupr_micro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.AUPR_Macro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.aupr_macro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Recall_Weighted:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.recall_weighted)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Recall_Micro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.recall_micro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Recall_Macro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.recall_macro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Precision_Weighted:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.precision_weighted)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Precision_Micro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.precision_micro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Precision_Macro:
            data_result = [self.training_result_repository.get_training_result(train_id, TrainingResultType.precision_macro)
                           for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[i.result_value for i in data_result],
                                  labels=[i.training_name for i in data_result])

        elif compare_plot_type == ComparePlotType.Details_Accuracy:

            data_result_details = [
                self.training_result_detail_repository.find_all_training_result_details(train_id)
                for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[[i.accuracy for i in r] for r in data_result_details],
                                  labels=[r[0].training_name for r in data_result_details if len(r) > 0])

        elif compare_plot_type == ComparePlotType.Details_F1_Score:

            data_result_details = [
                self.training_result_detail_repository.find_all_training_result_details(train_id)
                for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[[i.f1_score for i in r] for r in data_result_details],
                                  labels=[r[0].training_name for r in data_result_details if len(r) > 0])

        elif compare_plot_type == ComparePlotType.Details_AUC:

            data_result_details = [
                self.training_result_detail_repository.find_all_training_result_details(train_id)
                for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[[i.auc for i in r] for r in data_result_details],
                                  labels=[r[0].training_name for r in data_result_details if len(r) > 0])

        elif compare_plot_type == ComparePlotType.Details_AUPR:

            data_result_details = [
                self.training_result_detail_repository.find_all_training_result_details(train_id)
                for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[[i.aupr for i in r] for r in data_result_details],
                                  labels=[r[0].training_name for r in data_result_details if len(r) > 0])

        elif compare_plot_type == ComparePlotType.Details_Recall:

            data_result_details = [
                self.training_result_detail_repository.find_all_training_result_details(train_id)
                for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[[i.recall for i in r] for r in data_result_details],
                                  labels=[r[0].training_name for r in data_result_details if len(r) > 0])

        elif compare_plot_type == ComparePlotType.Details_Precision:

            data_result_details = [
                self.training_result_detail_repository.find_all_training_result_details(train_id)
                for train_id in train_ids]

            return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                  datas=[[i.precision for i in r] for r in data_result_details],
                                  labels=[r[0].training_name for r in data_result_details if len(r) > 0])

    @staticmethod
    def group_by_category(data):
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item.interaction_type].append(item)
        return grouped_data

    def stratified_sample(self, data: list[TrainingDrugInteractionDTO], test_size=0.1, min_samples_per_category=5) -> list[TrainingDrugInteractionDTO]:
        grouped_data = self.group_by_category(data)

        test_data = []

        for category, items in grouped_data.items():
            sample_size = max(int(len(items) * test_size), min_samples_per_category)

            if len(items) <= sample_size:
                test_data.extend(items)
            else:
                test_data.extend(random.sample(items, sample_size))

        return test_data

    # region Prepare data
    def prepare_drug_data(self, train_request: TrainRequestViewModel) -> list[TrainingDrugDataDTO]:

        drugs = self.drug_repository.find_all_active_drugs(True, True, True, True)

        if train_request.substructure_similarity:
            drugs = self.set_drug_similarity_training_values(drugs, train_request.substructure_similarity, Category.Substructure)

        if train_request.target_similarity:
            drugs = self.set_drug_similarity_training_values(drugs, train_request.target_similarity, Category.Target)

        if train_request.enzyme_similarity:
            drugs = self.set_drug_similarity_training_values(drugs, train_request.enzyme_similarity, Category.Enzyme)

        if train_request.pathway_similarity:
            drugs = self.set_drug_similarity_training_values(drugs, train_request.pathway_similarity, Category.Pathway)

        if train_request.description_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.description_embedding, TextType.Description)

        if train_request.indication_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.indication_embedding, TextType.Indication)

        if train_request.pharmacodynamics_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.pharmacodynamics_embedding, TextType.Pharmacodynamics)

        if train_request.mechanism_of_action_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.mechanism_of_action_embedding, TextType.MechanismOfAction)

        if train_request.toxicity_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.toxicity_embedding, TextType.Toxicity)

        if train_request.metabolism_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.metabolism_embedding, TextType.Metabolism)

        if train_request.absorption_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.absorption_embedding, TextType.Absorption)

        if train_request.half_life_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.half_life_embedding, TextType.HalfLife)

        if train_request.protein_binding_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.protein_binding_embedding, TextType.ProteinBinding)

        if train_request.route_of_elimination_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.route_of_elimination_embedding, TextType.RouteOfElimination)

        if train_request.volume_of_distribution_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.volume_of_distribution_embedding, TextType.VolumeOfDistribution)

        if train_request.clearance_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.clearance_embedding, TextType.Clearance)

        if train_request.classification_description_embedding:
            drugs = self.set_drug_embedding_training_values(drugs, train_request.classification_description_embedding, TextType.ClassificationDescription)

        return drugs

    def set_drug_similarity_training_values(self, drugs: list[TrainingDrugDataDTO], similarity_type: SimilarityType, category: Category) \
            -> list[TrainingDrugDataDTO]:

        print(f'Fetching {category.name} data!')

        if similarity_type != SimilarityType.Original:
            similarities = self.similarity_repository.find_all_active_similarity(similarity_type=similarity_type, category=category)

            similarities = sorted(similarities, key=lambda similarity: (similarity.drug_1, similarity.drug_2))

            similarities_by_drug_1 = defaultdict(list)
            for s in similarities:
                similarities_by_drug_1[s.drug_1].append(s)

            for d in drugs:
                values = {s.drug_2: s.value for s in similarities_by_drug_1.get(d.drug_id, [])}
                d.train_values.append(TrainingDrugTrainValuesDTO(category=category, values=values))

        else:
            similarities = self.get_original_data(category)
            for d in tqdm(drugs, f'Updating {category.name} data...'):
                if category != Category.Substructure:
                    d.train_values.append(TrainingDrugTrainValuesDTO(category=category, values=(similarities[d.drug_id] or [])))
                else:
                    value = next((s for k, s in similarities.items() if k == d.drug_id), None)

                    d.train_values.append(TrainingDrugTrainValuesDTO(category=category,
                                                                     values=smiles_helper.smiles_to_feature_matrix(value)))

                    d.train_values.append(TrainingDrugTrainValuesDTO(category=category,
                                                                     values=smiles_helper.smiles_to_adjacency_matrix(value)))

        return drugs

    def get_original_data(self, category: Category):
        if category == Category.Substructure:
            results = self.drug_repository.get_all_drug_smiles()

            return {r.id: r.smiles for r in results}

        elif category == Category.Target:
            results = self.target_repository.get_drug_target_as_feature()

            return {r.drug_id: r.features for r in results}

        elif category == Category.Pathway:
            results = self.pathway_repository.get_drug_pathway_as_feature()

            return {r.drug_id: r.features for r in results}

        elif category == Category.Enzyme:
            results = self.enzyme_repository.get_drug_enzyme_as_feature()

            return {r.drug_id: r.features for r in results}

        else:
            raise Exception(f'Unexpected category {category}')

    def set_drug_embedding_training_values(self, drugs: list[TrainingDrugDataDTO], embedding_type: EmbeddingType, text_type: TextType):
        print(f'Fetching {text_type.name}-{embedding_type.name} data!')
        embeddings = self.drug_embedding_repository.find_all_embedding_dict(embedding_type=embedding_type, text_type=text_type)

        embeddings = {k: self.validate_and_reshape(json.loads(v)) for k, v in tqdm(embeddings.items(), "Converting string to array...")}

        for d in tqdm(drugs, f"Fetching {text_type.name}-{embedding_type.name}...."):
            d.train_values.append(TrainingDrugTrainValuesDTO(category=embedding_helper.find_category(embedding_type, text_type),
                                                             values=embeddings.get(d.drug_id, [])))

        return drugs

    def prepare_interaction_data(self, train_request: TrainRequestViewModel, min_sample_count: int) -> list[TrainingDrugInteractionDTO]:

        print("Fetching Interactions!")
        interaction_data = self.drug_interaction_repository.find_training_interactions(min_sample_count, True, True, True, True)

        if train_request.interaction_description_embedding:

            embeddings = self.drug_embedding_repository.find_all_interaction_embedding_dict(embedding_type=train_request.interaction_description_embedding,
                                                                                            text_type=TextType.InteractionDescription)

            embeddings = {k: self.validate_and_reshape(json.loads(v)) for k, v in tqdm(embeddings.items(), "Converting string to array...")}

            for interaction in tqdm(interaction_data,
                                    f"Fetching {TextType.InteractionDescription.name}-{train_request.interaction_description_embedding.name}...."):
                interaction.interaction_description = embeddings.get(interaction.id, [])

        return interaction_data

    @staticmethod
    def validate_and_reshape(array):
        if len(array) != 1:
            raise ValueError("The length of the outer list is not 1.")
        return np.array(array).reshape(-1, len(array[0]))

    # endregion
