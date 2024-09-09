import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO

train_model = TrainModel.SimpleOneInput


class TrainPlan2(TrainPlanBase):

    @staticmethod
    def create_model(input_shape):
        input_data = layers.Input(shape=input_shape)
        x_data = layers.Dense(64, activation='relu')(input_data)
        x_data = layers.Dense(32, activation='relu')(x_data)
        return models.Model(inputs=input_data, outputs=x_data)

    def train(self, data: list[list[TrainingDataDTO]], train_id: int):
        x_train, x_test, y_train, y_test = super().split_train_test(data)

        input_models = []
        input_layers = []

        # Iterate over the datasets in 'data'
        for d in x_train:
            # # Extract the 'concat_values' from TrainingDataDTO for each TrainingDataDTO instance
            # concat_values = np.array([item.concat_values for item in d])

            # Create a model for the shape of the 'concat_values'
            model = self.create_model(d.shape[1:])  # Use the correct input shape for 'concat_values'
            input_models.append(model)
            input_layers.append(model.input)  # Store input layers for later use

        # Concatenate both models' outputs
        concatenated = layers.Concatenate()([model.output for model in input_models])

        # Add a fully connected layer before softmax
        x = layers.Dense(32, activation='relu')(concatenated)

        # Output layer with softmax activation
        output = layers.Dense(65, activation='softmax')(x)

        # Create the final model
        final_model = models.Model(inputs=input_layers, outputs=output)

        # Compile the model
        final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        final_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
        print('Training complete.')
