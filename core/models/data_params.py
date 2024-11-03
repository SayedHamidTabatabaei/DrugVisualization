class DataParams:
    def __init__(self, x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val if x_val is not None else x_test
        self.y_val = y_val if y_val is not None else y_test
        self.x_test = x_test
        self.y_test = y_test
