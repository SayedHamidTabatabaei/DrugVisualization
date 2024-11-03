from businesses.trains.layers.deep_ddi_layer import DeepDDIModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Deep_DDI


class DeepDDITrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)
        self.drug_channels: int = 256
        self.hidden_channels: int = 2048
        self.hidden_layers_num: int = 9
        self.dropout_prob: float = 0.2

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, train_id=parameters.train_id,
                                                                    padding=True, pca_generating=True)

        model = DeepDDIModel(hidden_size=self.hidden_channels,
                             hidden_layers_num=self.hidden_layers_num,
                             output_size=self.num_classes,
                             dropout_prob=self.dropout_prob)

        model.compile(optimizer="adam",
                      loss='binary_crossentropy',  # Multi-label classification
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_data=(x_test, y_test))

        result = self.calculate_evaluation_metrics(model, x_test, y_test)

        self.plot_accuracy(history, parameters.train_id)
        self.plot_loss(history, parameters.train_id)

        result.model_info = self.get_model_info(model)

        if parameters.interaction_data is not None:
            result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test)

        return result
