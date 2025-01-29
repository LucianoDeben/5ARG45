import os
import sys

import pandas as pd
import pytest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import run_tf_activity_inference


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
