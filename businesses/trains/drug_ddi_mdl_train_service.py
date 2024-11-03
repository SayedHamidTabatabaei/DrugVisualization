from tensorflow.keras import layers, Model, Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping

from businesses.trains.models.ddi_mdl_layer import DDIMDLLayer
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Drug_DDIMDL


class DrugDDIMDLTrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)
        self.droprate: float = 0.3

    def dnn(self, vector_size):
        train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
        train_in = layers.Dense(512, activation='relu')(train_input)
        train_in = layers.BatchNormalization()(train_in)
        train_in = layers.Dropout(self.droprate)(train_in)
        train_in = layers.Dense(256, activation='relu')(train_in)
        train_in = layers.BatchNormalization()(train_in)
        train_in = layers.Dropout(self.droprate)(train_in)
        train_in = layers.Dense(self.num_classes)(train_in)
        out = layers.Activation('softmax')(train_in)
        model = Model(inputs=train_input, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        results = []

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data, parameters.interaction_data, padding=True):

            model = None
            y_pred = None

            for idx, _ in enumerate(x_train):

                inputs = Input(shape=(x_train[idx].shape[1],), name='Inputlayer')
                ddi_layer = DDIMDLLayer()(inputs)

                model = Model(inputs=inputs, outputs=ddi_layer)

                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

                history = model.fit(x_train[idx], y_train, epochs=100, batch_size=128, validation_data=(x_test[idx], y_test), callbacks=early_stopping)

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

            results.append(result)

        return super().calculate_fold_results(results)
