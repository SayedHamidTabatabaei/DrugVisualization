import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical

from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO


class TrainPlanBase:
    category: TrainModel

    def __init__(self, category: TrainModel):
        self.category = category

    @staticmethod
    def split_train_test(data: list[list[TrainingDataDTO]]):
        # Prepare input for the model
        x_pairs = np.array([[item.concat_values for item in d] for d in data])

        # Example labels (replace this with your actual interaction data)
        y = np.array([item.interaction_type for item in data[0]])

        # Initialize empty lists
        x_train = [[] for _ in range(len(x_pairs))]
        x_test = [[] for _ in range(len(x_pairs))]

        y_train = []
        y_test = []

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_split.split(x_pairs[0], y):
            for i in range(len(x_pairs)):
                x_train[i] = x_pairs[i][train_index]
                x_test[i] = x_pairs[i][test_index]

            y_train, y_test = y[train_index], y[test_index]

        y_train = to_categorical(y_train, num_classes=65)
        y_test = to_categorical(y_test, num_classes=65)

        return x_train, x_test, y_train, y_test

    def save_image(self, train_id):
        pass

    def train(self, data: list[list[TrainingDataDTO]], train_id: int):
        pass
