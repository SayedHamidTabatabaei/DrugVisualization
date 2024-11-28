import numpy as np
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size=32, drop_remainder=False, padding=False, is_test=False, multiple_output:bool = False):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.padding = padding
        self.is_test = is_test
        self.multiple_output = multiple_output

    def __len__(self):
        if self.drop_remainder:
            return len(self.x_data[0]) // self.batch_size
        else:
            return int(np.ceil(len(self.x_data[0]) / self.batch_size))

    def __getitem__(self, index):

        start_idx = index * self.batch_size

        if self.padding:
            end_idx = min((index + 1) * self.batch_size, len(self.x_data[0]))
        else:
            end_idx = (index + 1) * self.batch_size

        if self.drop_remainder and end_idx > len(self.x_data[0]):
            raise IndexError

        batch_x = [data[start_idx:end_idx] for data in self.x_data]

        if self.multiple_output:
            batch_y = [data[start_idx:end_idx] for data in self.y_data]
        else:
            batch_y = self.y_data[start_idx:end_idx]

        if end_idx - start_idx < self.batch_size:
            padding_size = self.batch_size - (end_idx - start_idx)
            batch_x = [np.pad(x, ((0, padding_size),) + ((0, 0),) * (x.ndim - 1), mode='constant') for x in batch_x]
            batch_y = [np.pad(y, ((0, padding_size), (0, 0)), mode='constant') for y in batch_y]

        if self.is_test:
            return tuple([np.array(x) for x in batch_x]), np.array(batch_y[-1])

        if self.multiple_output:
           return tuple([np.array(x) for x in batch_x]), tuple([np.array(y) for y in batch_y])
        else:
            return tuple([np.array(x) for x in batch_x]), np.array(batch_y)
