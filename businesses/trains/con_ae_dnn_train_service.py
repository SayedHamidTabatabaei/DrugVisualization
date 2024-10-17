from tensorflow.keras.layers import Input, Dense
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Con_AE_DNN


class ConAeDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128

    def create_autoencoder(self, input_shape):
        input_layer = Input(input_shape)
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        return input_layer, encoded

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, padding=True, flat=True)

        input_dim = x_train.shape[1]  # Number of features per sample
        input_layer, encoded_model = self.create_autoencoder(input_shape=(input_dim,))

        # Define DNN layers after AutoEncoded
        x = Dense(256, activation='relu')(encoded_model)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)  # Softmax for multi-class classification

        full_model = Model(inputs=input_layer, outputs=output)

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=full_model,
                                     interactions=parameters.interaction_data)
