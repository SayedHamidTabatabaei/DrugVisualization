import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from spektral.layers import GATConv
from tensorflow.keras.layers import Input, Dense, Concatenate
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
from tqdm import tqdm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.category import Category
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.GAT_AE_Con_DNN


class GatAeConDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128

    def create_autoencoder(self, input_shape):
        input_layer = Input(input_shape)
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        return input_layer, encoded

    @staticmethod
    def prepare_smiles_data(data_entry):
        """
        Prepares SMILES data into graph format suitable for a GAT model.

        Parameters:
        - data_entry: A single entry containing SMILES string(s) from TrainingDataDTO.

        Returns:
        - A dictionary containing 'node_features' and 'adjacency_matrix'.
        """
        smiles_string = data_entry[0].values_1  # Access SMILES string from data entry
        mol = Chem.MolFromSmiles(smiles_string[0])

        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles_string}")

        # Create adjacency matrix
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)

        # Create node features (e.g., atomic number)
        node_features = []
        for atom in mol.GetAtoms():
            # Using atomic number as feature, you can add more features if necessary
            node_features.append([atom.GetAtomicNum()])

        # Convert node features to numpy array
        node_features = np.array(node_features, dtype=np.float32)

        return node_features, adjacency_matrix

    @staticmethod
    def create_gat_model_on_smiles(node_features, adjacency_matrix, num_heads=4, hidden_units=64, output_dim=32):
        """
        Creates a GAT model for processing SMILES data.

        Parameters:
        - node_features, adjacency_matrix: The input values for the SMILES data, which should be preprocessed graph data (node features and adjacency).
        - num_heads: The number of attention heads in the GAT layers.
        - hidden_units: The number of hidden units in the GAT layers.
        - output_dim: The output dimension for the dense encoding layer.

        Returns:
        - The encoded output from the GAT model.
        """

        # Define the input layers for node features and adjacency matrix
        node_input = Input(shape=(node_features.shape[1],), name="Node_Input")  # Shape: (num_nodes, feature_dim)
        adj_input = Input(shape=(node_features.shape[0],), name="Adj_Input")  # Shape: (num_nodes, num_nodes)

        # Define GAT layers
        x = GATConv(hidden_units, activation='relu', attn_heads=num_heads)([node_input, adj_input])
        x = GATConv(hidden_units, activation='relu', attn_heads=num_heads)([x, adj_input])

        # Dense layer to output the encoded representation
        encoded_output = Dense(output_dim, activation='relu')(x)

        # Create the GAT model
        gat_model = Model(inputs=[node_input, adj_input], outputs=encoded_output)

        return gat_model([node_features, adjacency_matrix])

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, padding=True)

        input_layers = []
        encoded_models = []

        for idx, d in tqdm(enumerate(x_train), "Creating Encoded Models..."):

            shapes = {tuple(s.shape) for s in d}

            if len(shapes) != 1:
                raise ValueError(f"Error: Multiple shapes found: {shapes}")

            if parameters.drug_data[0].train_values[idx].category != Category.Substructure:
                input_layer, encoded_model = self.create_autoencoder(input_shape=shapes.pop())
                input_layers.append(input_layer)
                encoded_models.append(encoded_model)
            else:
                input_layer = Input(shape=shapes.pop())
                node_features, adjacency_matrix = self.prepare_smiles_data(parameters.data[idx])  # Retrieve preprocessed SMILES graph data

                # Call the GAT model creation function
                gat_model_output = self.create_gat_model_on_smiles(node_features, adjacency_matrix)
                encoded_model = Dense(self.encoding_dim, activation='relu')(gat_model_output)
                input_layers.append(input_layer)
                encoded_models.append(encoded_model)

        concatenated = Concatenate()([encoded for encoded in encoded_models])

        # Define DNN layers after concatenation
        x = Dense(256, activation='relu')(concatenated)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)  # Softmax for multi-class classification

        # Create the full model
        full_model = Model(inputs=[input_layer for input_layer in input_layers], outputs=output)

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=full_model,
                                     interactions=parameters.interaction_data)
