from common.enums.loss_functions import LossFunctions


class TrainingParams:
    def __init__(self, train_id, optimizer='adam', loss: LossFunctions = LossFunctions.categorical_crossentropy, class_weight: bool = False, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.train_id = train_id
        self.optimizer = optimizer
        self.loss = loss
        self.class_weight = class_weight
        self.metrics = metrics
