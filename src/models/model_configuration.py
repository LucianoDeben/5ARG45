# src/models/model_configuration.py
import logging

from src.models.multimodal_models import MultimodalViabilityPredictor  # Adjust based on your model class

logger = logging.getLogger(__name__)

def configure_model(dataset, config: dict):
    """
    Configure and instantiate the model based on the dataset and configuration.

    Args:
        dataset: The training dataset to infer input dimensions from.
        config (dict): Configuration dictionary containing model hyperparameters.

    Returns:
        MultimodalModel: Instantiated model ready for training.
    """
    try:
        # Step 1: Infer input dimensions from the dataset
        sample = dataset[0]  # Assumes dataset.__getitem__ returns a dictionary
        transcriptomics_dim = sample["transcriptomics"].shape[0]
        molecular_dim = sample["molecular"].shape[0]

        # Step 2: Retrieve hyperparameters from config
        model_params = {
            "transcriptomics_input_dim": transcriptomics_dim,
            "molecular_input_dim": molecular_dim,
            "hidden_dims": config["model"]["hidden_dims"],
            "dropout": config["model"]["dropout"],
        }

        # Step 3: Instantiate the model
        model = MultimodalViabilityPredictor(**model_params)

        logger.info(f"Model configured with parameters: {model_params}")
        return model

    except Exception as e:
        logger.error(f"Error in model configuration: {str(e)}")
        raise