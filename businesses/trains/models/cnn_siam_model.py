import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, ReLU, MaxPooling1D
from tensorflow.keras.models import Model, Sequential

from businesses.trains.models.train_base_model import TrainBaseModel
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class CNNSiamModel(Model):
    def __init__(self, num_classes: int, vector_size: int = 936):
        super(CNNSiamModel, self).__init__()
        self.num_classes = num_classes
        self.vector_size = vector_size

        self.conv1 = Conv1D(64, 3, use_bias=False, padding='valid', name='conv1')
        self.conv2 = Conv1D(128, 3, use_bias=False, padding='valid', name='conv2')
        self.conv3_1 = Conv1D(128, 3, padding='same', use_bias=False, name='conv3_1')
        self.conv3_2 = Conv1D(128, 3, padding='same', use_bias=False, name='conv3_2')
        self.conv4 = Conv1D(256, 3, use_bias=False, name='conv4')
        self.bn4 = BatchNormalization(name='batch_norm_4')

        # Fully Connected Layers
        self.fc = Sequential([
            Dense(2048, activation=None, name='fc1'),
            ReLU(name='relu1'),
            Dense(256, activation=None, name='fc2'),
            ReLU(name='relu2'),
            Dense(self.num_classes, activation=None, name='fc3')
        ], name='fully_connected')

    def encode(self, x):
        # Encoding path
        x = tf.nn.relu(self.conv1(x))
        res_bef = tf.nn.relu(self.conv2(x))

        x = tf.nn.relu(self.conv3_1(res_bef))
        res_aft = tf.nn.relu(self.conv3_2(x))

        # Residual connection
        x = res_aft + res_bef
        x = tf.nn.relu(self.bn4(self.conv4(x)))
        x = MaxPooling1D(pool_size=2)(x)

        return x

    def call(self, inputs, **kwargs):
        # Split input into two parts
        x = tf.reshape(inputs, [-1, 3, self.vector_size])
        x1 = x[:, :, :self.vector_size // 2]
        x2 = x[:, :, self.vector_size // 2:]

        # Encode both parts
        x1 = self.encode(x1)
        x2 = self.encode(x2)

        # Add embeddings
        x = x1 + x2

        # Flatten for fully connected layers
        x = tf.reshape(x, [-1, 256 * 283])

        # Pass through fully connected layers
        y = self.fc(x)
        return y


class CNNSiamTrainModel(TrainBaseModel):
    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO]):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:
        model = CNNSiamModel(self.num_classes)

        model.compile(optimizer="adam", loss="focal_loss", metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto')

        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), callbacks=early_stopping)

        self.save_plots(history, self.train_id)

        y_pred = model.predict(x_test)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
