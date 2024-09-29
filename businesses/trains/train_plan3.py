import tensorflow as tf
from tensorflow.keras import layers, models

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.SumSoftmaxOutputs


class TrainPlan3(TrainPlanBase):

    def create_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))  # Softmax output
        return model

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(data)

        models_list = [self.create_model(d[0].shape) for d in x_train]

        inputs = [tf.keras.Input(shape=d[0].shape) for d in x_train]

        softmax_outputs = [model(input_layer) for model, input_layer in zip(models_list, inputs)]

        summed_output = tf.keras.layers.Add()(softmax_outputs)

        final_model = tf.keras.Model(inputs=inputs, outputs=summed_output)
        final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        final_model.summary()

        x_train_ragged, x_test_ragged = super().create_input_tensors(x_train, x_test)

        print('Fit data!')
        history = final_model.fit(x_train_ragged, y_train, epochs=50, batch_size=256,
                                  validation_data=(x_test_ragged, y_test))

        evaluations = super().calculate_evaluation_metrics(final_model, x_test_ragged, y_test)

        super().plot_accuracy(history, parameters.train_id)
        super().plot_loss(history, parameters.train_id)

        # super().plot_accuracy_radial([item.accuracy for item in evaluations.training_result_details], parameters.train_id)
        # super().plot_f1_score_radial([item.f1_score for item in evaluations.training_result_details], parameters.train_id)
        # super().plot_auc_radial([item.auc for item in evaluations.training_result_details], parameters.train_id)

        return evaluations

