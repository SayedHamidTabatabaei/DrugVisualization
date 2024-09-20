import numpy as np

from businesses.reduction_services.reduction_base_service import ReductionBaseService
from common.enums.reduction_category import ReductionCategory
from core.domain.reduction_data import ReductionData

from core.models.reduction_parameter_model import ReductionParameterModel

from tensorflow.keras import layers, models

reduction_category = ReductionCategory.AutoEncoder_Max


class AutoEncoderReductionService(ReductionBaseService):

    def __init__(self, category: ReductionCategory, encoding_dim):
        super().__init__(category)
        self.encoding_dim = encoding_dim

    def reduce(self, parameters: ReductionParameterModel, data: dict) -> list[ReductionData]:

        values = np.array([value for key, value in data.items()])

        input_dim = values.shape[1]
        input_value = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_value)
        encoder = models.Model(input_value, encoded)
        autoencoder = models.Model(input_value, layers.Dense(input_dim, activation='sigmoid')(encoded))
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(values, values, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

        encoded = encoder.predict(values)

        return [ReductionData(drug_id=key,
                              similarity_type=parameters.similarity_type,
                              category=parameters.category,
                              reduction_category=self.category,
                              reduction_values=str(encoded[idx].tolist()),
                              has_enzyme=True,
                              has_pathway=True,
                              has_target=True,
                              has_smiles=True) for idx, (key, values) in enumerate(data.items())]
