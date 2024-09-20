from tensorflow.keras import layers, models
from tqdm import tqdm

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO

train_model = TrainModel.JoinSimplesBeforeSoftmax


class TrainPlan2(TrainPlanBase):

    @staticmethod
    def create_model(input_shape):
        input_data = layers.Input(shape=input_shape)
        x_data = layers.Dense(64, activation='relu')(input_data)
        x_data = layers.Dense(32, activation='relu')(x_data)
        return models.Model(inputs=input_data, outputs=x_data)

    def train(self, data: list[list[TrainingDataDTO]], train_id: int) -> TrainingResultSummaryDTO:

        print('split data!')
        x_train, x_test, y_train, y_test = super().split_train_test(data)

        input_models = []
        input_layers = []

        for d in tqdm(x_train, "Creating Models..."):
            shapes = {tuple(s.shape) for s in d}

            if len(shapes) != 1:
                raise ValueError(f"Error: Multiple shapes found: {shapes}")

            # Create a model for the shape of the 'concat_values'
            model = self.create_model(shapes.pop())  # Use the correct input shape for 'concat_values'
            input_models.append(model)
            input_layers.append(model.input)  # Store input layers for later use

        # Concatenate both models' outputs
        concatenated = layers.Concatenate()([model.output for model in input_models])

        # Add a fully connected layer before softmax
        x = layers.Dense(32, activation='relu')(concatenated)

        # Output layer with softmax activation
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        # Create the final model
        final_model = models.Model(inputs=input_layers, outputs=output)

        # Compile the model
        final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print('Ragged!')
        x_train_ragged, x_test_ragged = super().create_input_tensors(x_train, x_test)

        # Pass RaggedTensors to the model
        print('Fit data!')
        history = final_model.fit(x_train_ragged, y_train, epochs=20, batch_size=32,
                                  validation_data=(x_test_ragged, y_test))

        evaluations = super().calculate_evaluation_metrics(final_model, x_test_ragged, y_test)

        super().plot_accuracy(history, train_id)
        super().plot_loss(history, train_id)

        # super().plot_accuracy_radial([item.accuracy for item in evaluations.training_result_details], train_id)
        # super().plot_f1_score_radial([item.f1_score for item in evaluations.training_result_details], train_id)
        # super().plot_auc_radial([item.auc for item in evaluations.training_result_details], train_id)

        return evaluations
