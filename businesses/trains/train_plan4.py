from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tqdm import tqdm

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO

train_model = TrainModel.AutoEncoderWithDNN


class TrainPlan4(TrainPlanBase):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128

    def create_autoencoder(self, input_shape):
        input_layer = Input(input_shape)
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        return input_layer, encoded

    def train(self, data: list[list[TrainingDataDTO]], train_id: int) -> TrainingResultSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(data)

        input_layers = []
        encoded_models = []

        for d in tqdm(x_train, "Creating Encoded Models..."):

            shapes = {tuple(s.shape) for s in d}

            if len(shapes) != 1:
                raise ValueError(f"Error: Multiple shapes found: {shapes}")

            input_layer, encoded_model = self.create_autoencoder(shapes.pop())
            input_layers.append(input_layer)
            encoded_models.append(encoded_model)

        concatenated = Concatenate()([encoded for encoded in encoded_models])

        # Define DNN layers after concatenation
        x = Dense(256, activation='relu')(concatenated)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)  # Softmax for multi-class classification

        # Create the full model
        full_model = Model(inputs=[input_layer for input_layer in input_layers], outputs=output)
        full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x_train_ragged, x_test_ragged = super().create_ragged_tensors(x_train, x_test)

        print('Fit data!')
        history = full_model.fit(x_train_ragged, y_train, epochs=50, batch_size=256,
                                 validation_data=(x_test_ragged, y_test))

        evaluations = super().calculate_evaluation_metrics(full_model, x_test_ragged, y_test)

        super().plot_accuracy(history, train_id)
        super().plot_loss(history, train_id)

        super().plot_accuracy_radial([item.accuracy for item in evaluations.training_result_details], train_id)
        super().plot_f1_score_radial([item.f1_score for item in evaluations.training_result_details], train_id)
        super().plot_auc_radial([item.auc for item in evaluations.training_result_details], train_id)

        return evaluations
