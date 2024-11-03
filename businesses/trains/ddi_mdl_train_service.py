# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping

from businesses.trains.models.ddi_mdl_layer import DDIMDLLayer
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.DDIMDL


class DDIMDLTrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        model = None
        y_pred = None

        x_train, x_val, x_test, y_train, y_val, y_test = super().split_train_val_test(parameters.drug_data, parameters.interaction_data,
                                                                                      train_id=parameters.train_id,
                                                                                      padding=True, pca_generating=True,
                                                                                      pca_components=len(parameters.drug_data))

        for idx, _ in enumerate(x_train):

            model = DDIMDLLayer()

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

            history = model.fit(x_train[idx], y_train, epochs=100, batch_size=128, validation_data=(x_val[idx], y_val), callbacks=early_stopping)

            self.plot_accuracy(history, f"{parameters.train_id}_{idx}")
            self.plot_loss(history, f"{parameters.train_id}_{idx}")

            if y_pred is None:
                y_pred = model.predict(x_test[idx])
            else:
                y_pred += model.predict(x_test[idx])

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if parameters.interaction_data is not None:
            result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test)

        return result
