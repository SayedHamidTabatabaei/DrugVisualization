from tensorflow.keras import layers, Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping

from businesses.trains.models.train_base_model import TrainBaseModel
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class DDIMDLModel(Model):
    def __init__(self, droprate=0.3, num_classes=65):

        super(DDIMDLModel, self).__init__()

        self.input_layer = layers.Dense(512, activation='relu')
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(droprate)

        self.hidden_layer = layers.Dense(256, activation='relu')
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(droprate)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.hidden_layer(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        return self.output_layer(x)

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']

        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)


class DDIMDLTrainModel(TrainBaseModel):
    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO]):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:
        model = None
        y_pred = None

        for idx, _ in enumerate(x_train):

            model = DDIMDLModel()
            model.compile_model()

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

            history = model.fit(x_train[idx], y_train, epochs=100, batch_size=128, validation_data=(x_val[idx], y_val), callbacks=early_stopping)

            self.save_plots(history, f"{self.train_id}_{idx}")

            if y_pred is None:
                y_pred = model.predict(x_test[idx])
            else:
                y_pred += model.predict(x_test[idx])

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
