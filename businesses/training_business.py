import base64
import io
import mimetypes
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
from injector import inject
from tqdm import tqdm

from businesses import reduction_business
from businesses.base_business import BaseBusiness
from businesses.trains.train_instances import get_instance
from common.enums.category import Category
from common.enums.compare_plot_type import ComparePlotType
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType
from common.enums.train_models import TrainModel
from common.helpers import math_helper
from core.domain.training_result_detail import TrainingResultDetail
from core.models.train_request_model import TrainRequestModel
from core.repository_models.compare_plot_dto import ComparePlotDTO
from core.repository_models.interaction_dto import InteractionDTO
from core.repository_models.reduction_data_dto import ReductionDataDTO
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from core.repository_models.training_result_dto import TrainingResultDTO
from core.view_models.image_info_view_model import ImageInfoViewModel
from core.view_models.training_history_details_view_model import TrainingHistoryDetailsViewModel
from core.view_models.training_history_view_model import TrainingHistoryViewModel
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository
from infrastructure.repositories.training_result_detail_repository import TrainingResultDetailRepository
from infrastructure.repositories.training_result_repository import TrainingResultRepository


def map_training_data(interactions: list[InteractionDTO], reductions: list[ReductionDataDTO]) -> list[TrainingDataDTO]:
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
                                          interaction_type=interaction.interaction_type)
        training_data.append(training_entity)

    return training_data


def map_training_history_view_model(results: list[TrainingResultDTO]) -> list[TrainingHistoryViewModel]:
    return [TrainingHistoryViewModel(
        id=item.id,
        train_model=item.train_model.name,
        training_conditions=item.training_conditions,
        accuracy=item.accuracy,
        f1_score=item.f1_score,
        auc=item.auc,
        aupr=item.aupr,
        loss=item.loss,
        execute_time=item.execute_time
    ) for item in results]


def map_training_history_details_view_model(results: list[TrainingResultDetailDTO]) \
        -> list[TrainingHistoryDetailsViewModel]:
    return [TrainingHistoryDetailsViewModel(
        training_label=item.training_label,
        accuracy=item.accuracy,
        f1_score=item.f1_score,
        auc=item.auc,
        aupr=item.aupr
    ) for item in results]


colors = list(mcolors.TABLEAU_COLORS.values())


class TrainingBusiness(BaseBusiness):
    @inject
    def __init__(self, reduction_repository: ReductionDataRepository,
                 training_result_repository: TrainingResultRepository,
                 training_result_detail_repository: TrainingResultDetailRepository):
        BaseBusiness.__init__(self)
        self.reduction_repository = reduction_repository
        self.training_result_repository = training_result_repository
        self.training_result_detail_repository = training_result_detail_repository

    def train(self, train_request: TrainRequestModel):
        instance = get_instance(train_request.train_model)

        data = self.prepare_data(train_request)

        train_id = self.training_result_repository.insert(train_request.train_model, train_request.to_json())

        print('Start training...')
        training_result = instance.train(data, train_id)

        print('Updating...')
        self.training_result_repository.update(train_id,
                                               training_result.f1_score,
                                               training_result.accuracy,
                                               training_result.loss,
                                               training_result.auc,
                                               training_result.aupr)

        train_result_details = [TrainingResultDetail(training_result_id=train_id,
                                                     training_label=item.training_label,
                                                     f1_score=item.f1_score,
                                                     accuracy=item.accuracy,
                                                     auc=item.auc,
                                                     aupr=item.aupr) for item in
                                training_result.training_result_details]

        self.training_result_detail_repository.insert_batch_check_duplicate(train_result_details)

    def get_training_data(self, similarity_type: SimilarityType, category: Category,
                          reduction_category: ReductionCategory) -> list[TrainingDataDTO]:
        interactions = self.reduction_repository.find_interactions(similarity_type, category, reduction_category,
                                                                   True, True, True, True)

        reductions = self.reduction_repository.find_reductions(reduction_category, similarity_type, category,
                                                               True, True, True, True,
                                                               0, 100000)

        results = map_training_data(interactions, reductions)

        if not results:
            raise Exception(f"No training data found for {similarity_type} {category} {reduction_category}")

        return results

    def prepare_data(self, train_request):

        data = []

        if train_request.substructure_similarity and train_request.substructure_reduction:
            print('Fetching Substructure data!')
            data.append(self.get_training_data(train_request.substructure_similarity, Category.Substructure,
                                               train_request.substructure_reduction))

        if train_request.target_similarity and train_request.target_reduction:
            print('Fetching Target data!')
            data.append(self.get_training_data(train_request.target_similarity, Category.Target,
                                               train_request.target_reduction))

        if train_request.enzyme_similarity and train_request.enzyme_reduction:
            print('Fetching Enzyme data!')
            data.append(self.get_training_data(train_request.enzyme_similarity, Category.Enzyme,
                                               train_request.enzyme_reduction))

        if train_request.pathway_similarity and train_request.pathway_reduction:
            print('Fetching Pathway data!')
            data.append(self.get_training_data(train_request.pathway_similarity, Category.Pathway,
                                               train_request.pathway_reduction))

        if train_request.description_embedding and train_request.description_reduction:
            print('Fetching Description data!')
            category = reduction_business.find_category(train_request.description_embedding, TextType.Description)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.description_reduction))

        if train_request.indication_embedding and train_request.indication_reduction:
            print('Fetching Indication data!')
            category = reduction_business.find_category(train_request.indication_embedding, TextType.Indication)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.indication_reduction))

        if train_request.pharmacodynamics_embedding and train_request.pharmacodynamics_reduction:
            print('Fetching Pharmacodynamics data!')
            category = reduction_business.find_category(train_request.pharmacodynamics_embedding,
                                                        TextType.Pharmacodynamics)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.pharmacodynamics_reduction))

        if train_request.mechanism_of_action_embedding and train_request.mechanism_of_action_reduction:
            print('Fetching Mechanism of action data!')
            category = reduction_business.find_category(train_request.mechanism_of_action_embedding,
                                                        TextType.MechanismOfAction)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.mechanism_of_action_reduction))

        if train_request.toxicity_embedding and train_request.toxicity_reduction:
            print('Fetching Toxicity data!')
            category = reduction_business.find_category(train_request.toxicity_embedding, TextType.Toxicity)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.toxicity_reduction))

        if train_request.metabolism_embedding and train_request.metabolism_reduction:
            print('Fetching Metabolism data!')
            category = reduction_business.find_category(train_request.metabolism_embedding, TextType.Metabolism)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.metabolism_reduction))

        if train_request.absorption_embedding and train_request.absorption_reduction:
            print('Fetching Absorption data!')
            category = reduction_business.find_category(train_request.absorption_embedding, TextType.Absorption)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.absorption_reduction))

        if train_request.half_life_embedding and train_request.half_life_reduction:
            print('Fetching Half life data!')
            category = reduction_business.find_category(train_request.half_life_embedding, TextType.HalfLife)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.half_life_reduction))

        if train_request.protein_binding_embedding and train_request.protein_binding_reduction:
            print('Fetching Protein binding data!')
            category = reduction_business.find_category(train_request.protein_binding_embedding,
                                                        TextType.ProteinBinding)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.protein_binding_reduction))

        if train_request.route_of_elimination_embedding and train_request.route_of_elimination_reduction:
            print('Fetching Route of elimination data!')
            category = reduction_business.find_category(train_request.route_of_elimination_embedding,
                                                        TextType.RouteOfElimination)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.route_of_elimination_reduction))

        if train_request.volume_of_distribution_embedding and train_request.volume_of_distribution_reduction:
            print('Fetching Volume of distribution data!')
            category = reduction_business.find_category(train_request.volume_of_distribution_embedding,
                                                        TextType.VolumeOfDistribution)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.volume_of_distribution_reduction))

        if train_request.clearance_embedding and train_request.clearance_reduction:
            print('Fetching Clearance data!')
            category = reduction_business.find_category(train_request.clearance_embedding, TextType.Clearance)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.clearance_reduction))

        if train_request.classification_description_embedding and train_request.classification_description_reduction:
            print('Fetching Classification Description data!')
            category = reduction_business.find_category(train_request.classification_description_embedding,
                                                        TextType.ClassificationDescription)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.classification_description_reduction))

        return data

    def get_history(self, train_model: TrainModel, start: int, length: int):

        total_number = self.training_result_repository.get_training_result_count(train_model)

        results = self.training_result_repository.find_all_training_result(train_model, start, length)

        return map_training_history_view_model(results), total_number[0]

    def get_history_details(self, train_result_id: int):

        results = self.training_result_detail_repository.find_all_training_result_details(train_result_id)

        return map_training_history_details_view_model(results)

    def get_history_conditions(self, train_result_id: int):

        result = self.training_result_repository.get_training_result_by_id(train_result_id)

        return result.training_conditions

    def get_history_plots(self, train_result_id: int) -> list[ImageInfoViewModel]:

        image_folder = f'training_plots/{train_result_id}'

        image_files: list[ImageInfoViewModel] = []

        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)

            if mimetypes.guess_type(file_path)[0] and 'image' in mimetypes.guess_type(file_path)[0]:
                file_name, _ = os.path.splitext(file)

                image_files.append(ImageInfoViewModel(path=f"{file_path}", name=file_name))

        return image_files

    def get_comparing_plots(self, train_result_ids: list[int]) -> list[ImageInfoViewModel]:

        images: list[ImageInfoViewModel] = []
        plot_list = self.fill_plot_list(train_result_ids)

        for plot in plot_list:
            plt = self.generate_plot(plot)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            plt.close()

            image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')

            images.append(ImageInfoViewModel(path=image_base64, name=plot.compare_plot_type.name))

        return images

    def get_comparing_plot(self, train_result_ids: list[int], compare_plot_type: ComparePlotType) \
            -> ImageInfoViewModel:

        plot_info = self.get_plot_info(compare_plot_type, train_result_ids)

        plt = self.generate_plot(plot_info)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')

        return ImageInfoViewModel(path=image_base64, name=plot_info.compare_plot_type.name)

    def generate_plot(self, plot: ComparePlotDTO):
        match plot.compare_plot_type:
            case (ComparePlotType.Accuracy | ComparePlotType.Loss | ComparePlotType.AUC | ComparePlotType.AUPR |
                  ComparePlotType.F1_Score):
                return self.plot_bar(plot)
            case (ComparePlotType.Details_Accuracy | ComparePlotType.Details_AUC | ComparePlotType.Details_AUPR |
                  ComparePlotType.Details_F1_Score):
                return self.plot_radial(plot)

    @staticmethod
    def plot_bar(plot_data: ComparePlotDTO):
        plt.figure()

        x = range(len(plot_data.datas))

        bars = plt.bar(x, plot_data.datas, color=colors[:len(plot_data.datas)], label=plot_data.labels)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.5f}', ha='center', va='bottom')

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
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, data in enumerate(plot_data.datas):
            data = np.concatenate((data, [data[0]]))

            ax.plot(angles, data, linewidth=2, linestyle='solid', label=plot_data.labels[i])

        category_labels = [i for i in range(num_categories)]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(category_labels, fontsize=8)

        ax.set_ylim(min([min(d) for d in plot_data.datas]) * 0.9,
                    math_helper.round_up(max([max(d) for d in plot_data.datas]), 1))

        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        plt.title(plot_data.compare_plot_type.name)

        return plt

    def fill_plot_list(self, train_result_ids: list[int]):
        data_result = [self.training_result_repository.get_training_result_by_id(train_result_id)
                       for train_result_id in train_result_ids]

        data_result_details = [self.training_result_detail_repository.find_all_training_result_details(train_result_id)
                               for train_result_id in train_result_ids]

        plot_list = [
            ComparePlotDTO(compare_plot_type=ComparePlotType.Accuracy,
                           datas=[i.accuracy for i in data_result],
                           labels=[i.id for i in data_result]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.Loss,
                           datas=[i.loss for i in data_result],
                           labels=[i.id for i in data_result]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.AUC,
                           datas=[i.auc for i in data_result],
                           labels=[i.id for i in data_result]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.AUPR,
                           datas=[i.aupr for i in data_result],
                           labels=[i.id for i in data_result]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.F1_Score,
                           datas=[i.f1_score for i in data_result],
                           labels=[i.id for i in data_result]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.Details_Accuracy,
                           datas=[[i.accuracy for i in r] for r in data_result_details],
                           labels=[r[0].training_result_id for r in data_result_details]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.Details_AUC,
                           datas=[[i.auc for i in r] for r in data_result_details],
                           labels=[r[0].training_result_id for r in data_result_details]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.Details_AUPR,
                           datas=[[i.aupr for i in r] for r in data_result_details],
                           labels=[r[0].training_result_id for r in data_result_details]),
            ComparePlotDTO(compare_plot_type=ComparePlotType.Details_F1_Score,
                           datas=[[i.f1_score for i in r] for r in data_result_details],
                           labels=[r[0].training_result_id for r in data_result_details])
        ]

        return plot_list

    def get_plot_info(self, compare_plot_type: ComparePlotType, train_result_ids: list[int]) -> ComparePlotDTO:
        match compare_plot_type:
            case ComparePlotType.Accuracy:
                data_result = [self.training_result_repository.get_training_result_by_id(train_result_id)
                               for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.accuracy for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Loss:
                data_result = [self.training_result_repository.get_training_result_by_id(train_result_id)
                               for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.loss for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUC:
                data_result = [self.training_result_repository.get_training_result_by_id(train_result_id)
                               for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.auc for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.AUPR:
                data_result = [self.training_result_repository.get_training_result_by_id(train_result_id)
                               for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.aupr for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.F1_Score:
                data_result = [self.training_result_repository.get_training_result_by_id(train_result_id)
                               for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[i.f1_score for i in data_result],
                                      labels=[i.id for i in data_result])

            case ComparePlotType.Details_Accuracy:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_result_id)
                    for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.accuracy for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details])

            case ComparePlotType.Details_AUC:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_result_id)
                    for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.auc for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details])

            case ComparePlotType.Details_AUPR:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_result_id)
                    for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.aupr for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details])

            case ComparePlotType.Details_F1_Score:

                data_result_details = [
                    self.training_result_detail_repository.find_all_training_result_details(train_result_id)
                    for train_result_id in train_result_ids]

                return ComparePlotDTO(compare_plot_type=compare_plot_type,
                                      datas=[[i.f1_score for i in r] for r in data_result_details],
                                      labels=[r[0].training_result_id for r in data_result_details])
