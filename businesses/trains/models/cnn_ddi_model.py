from tensorflow.keras import layers, Model

from businesses.trains.models.train_base_model import TrainBaseModel
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class CNNDDIModel(Model):
    def __init__(self, num_classes: int):
        super(CNNDDIModel, self).__init__()
        self.num_classes = num_classes

        # Convolutional Layers for each input
        self.conv1 = layers.Conv2D(64, (3, 1), padding='same')
        self.conv2 = layers.Conv2D(128, (3, 1), padding='same')
        self.conv3_1 = layers.Conv2D(128, (3, 1), padding='same')
        self.conv3_2 = layers.Conv2D(128, (3, 1), padding='same')
        self.conv4 = layers.Conv2D(256, (3, 1), padding='same')

        # Fully Connected Layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256)
        self.fc2 = layers.Dense(self.num_classes)  # Assuming 65 DDI types

        # Leaky ReLU Activation
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, **kwargs):
        # Process each input independently through the convolutional layers
        x = self.leaky_relu(self.conv1(inputs))
        x = self.leaky_relu(self.conv2(x))

        # Residual connection
        identity = x
        x = self.leaky_relu(self.conv3_1(x))
        x = self.conv3_2(x)
        x = layers.add([x, identity])  # Add residual connection
        x = self.leaky_relu(x)

        x = self.leaky_relu(self.conv4(x))
        x = self.flatten(x)

        # Fully connected layers
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNDDITrainModel(TrainBaseModel):
    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO]):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:
        model = CNNDDIModel(self.num_classes)

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))

        self.save_plots(history, self.train_id)

        y_pred = model.predict(x_test)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
