import os
import sys

import pandas as pd
import pytest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import run_tf_activity_inference, split_data


@pytest.fixture
def sample_data():
    """Creates a sample gene expression dataset with metadata."""
    metadata = {
        "cell_mfc_name": ["cell1", "cell2", "cell3"],
        "viability": [0.8, 0.6, 0.9],
        "pert_dose": [10, 20, 30],
    }

    gene_expression = {
        "TF1": [1.2, 0.5, 0.8],
        "TF2": [0.9, 1.1, 0.7],
        "TF3": [1.4, 1.2, 0.6],
    }

    X = pd.DataFrame({**metadata, **gene_expression})
    return X


@pytest.fixture
def sample_network():
    """Creates a sample regulatory network."""
    return pd.DataFrame(
        {"source": ["TF1", "TF2"], "target": ["TF2", "TF3"], "weight": [0.5, 0.7]}
    )


@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing regression splits."""
    data = {
        "feature1": range(100),
        "feature2": range(100, 200),
        "cell_mfc_name": [f"group_{i%5}" for i in range(100)],
        "viability": [
            x * 0.5 + 10 for x in range(100)
        ],  # Continuous target for regression
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Creates a sample configuration dictionary for splitting."""
    return {
        "preprocess": {
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
        }
    }


@pytest.mark.run_tf_activity_inference
def test_run_tf_activity_inference(sample_data, sample_network):
    """Test valid execution of TF activity inference."""
    result = run_tf_activity_inference(sample_data, sample_network, min_n=1)
    assert isinstance(result, pd.DataFrame)
    assert "cell_mfc_name" in result.columns
    assert result.shape[0] == 3  # Should retain original rows


@pytest.mark.run_tf_activity_inference
def test_missing_metadata_columns():
    """Test handling of missing metadata columns."""
    data = pd.DataFrame({"TF1": [1.2, 0.5], "TF2": [0.9, 1.1]})
    network = pd.DataFrame({"source": ["TF1"], "target": ["TF2"], "weight": [0.5]})

    with pytest.raises(ValueError, match="Missing expected metadata columns"):
        run_tf_activity_inference(data, network)


@pytest.mark.run_tf_activity_inference
def test_no_shared_genes(sample_data):
    """Test when there are no shared genes between network and expression data."""
    network = pd.DataFrame({"source": ["TFX"], "target": ["TFY"], "weight": [0.5]})

    with pytest.raises(ValueError, match="No shared genes found"):
        run_tf_activity_inference(sample_data, network)


@pytest.mark.split_data
def test_split_data_valid(sample_dataframe, sample_config):
    """Test splitting with valid input for regression."""
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        sample_dataframe, sample_config, target_name="viability", stratify_by=None
    )

    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0
    assert len(y_train) > 0 and len(y_val) > 0 and len(y_test) > 0
    assert (
        "cell_mfc_name" not in X_train.columns
    )  # Should be removed when stratify_by is None


@pytest.mark.split_data
def test_split_data_invalid_ratios(sample_dataframe):
    """Test if split_data raises an error when ratios do not sum to 1."""
    invalid_config = {
        "preprocess": {"train_ratio": 0.6, "val_ratio": 0.3, "test_ratio": 0.2}
    }
    with pytest.raises(ValueError, match="Ratios must sum to 1"):
        split_data(sample_dataframe, invalid_config, target_name="viability")


@pytest.mark.split_data
def test_split_data_stratification(sample_dataframe, sample_config):
    """Test stratified split with valid stratify_by column."""
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        sample_dataframe,
        sample_config,
        target_name="viability",
        stratify_by="cell_mfc_name",
    )

    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0
    assert "cell_mfc_name" not in X_train.columns  # Should be removed after split


@pytest.mark.split_data
def test_split_data_invalid_stratify_column(sample_dataframe, sample_config):
    """Test if split_data raises an error when stratify_by column does not exist."""
    with pytest.raises(KeyError):
        split_data(
            sample_dataframe,
            sample_config,
            target_name="viability",
            stratify_by="non_existent_column",
        )


def test_split_data_minimum_group_requirement(sample_dataframe, sample_config):
    """Test that split_data raises an error when stratifying with insufficient unique groups."""
    sample_dataframe["cell_mfc_name"] = "only_one_group"  # Only one unique value
    with pytest.raises(ValueError, match="Need at least 2 groups"):
        split_data(
            sample_dataframe,
            sample_config,
            target_name="viability",
            stratify_by="cell_mfc_name",
        )
