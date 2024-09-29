import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.ContactDataWithOneDNN


class TrainPlan5(TrainPlanBase):

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:
        x_train, x_test, y_train, y_test = super().split_train_test(data)

        x_train_concat_data = tf.concat([x for x in tqdm(x_train, "Concat train data")], axis=1)
        x_test_concat_data = tf.concat([x for x in tqdm(x_test, "Concat test data")], axis=1)

        input_shape = x_train_concat_data.shape[1]

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_shape,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

        print('Fit data!')
        history = model.fit(x_train_concat_data, y_train, epochs=50, batch_size=256,
                            validation_data=(x_test_concat_data, y_test))

        evaluations = super().calculate_evaluation_metrics(model, x_test_concat_data, y_test)

        super().plot_accuracy(history, parameters.train_id)
        super().plot_loss(history, parameters.train_id)

        # super().plot_accuracy_radial([item.accuracy for item in evaluations.training_result_details], parameters.train_id)
        # super().plot_f1_score_radial([item.f1_score for item in evaluations.training_result_details], parameters.train_id)
        # super().plot_auc_radial([item.auc for item in evaluations.training_result_details], parameters.train_id)

        return evaluations
