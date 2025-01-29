import os
import sys
from io import StringIO
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.utils import create_dataloader, load_config, load_sampled_data, sanity_check


class SimpleModel(nn.Module):
    """A simple linear model for testing purposes."""

    def __init__(self, input_dim: int):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def sample_torch_data():
    """Create a small Pytorch dataset for testing."""
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)


@pytest.fixture
def sample_pandas_data():
    """Creates a small Pandas feature dataset and target variable for testing."""
    X = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0, 4.0, 5.0], "feature2": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )
    y = pd.Series([1, 0, 1, 0, 1])
    return X, y


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(input_dim=5)


@pytest.fixture
def sample_config():
    """Creates a temporary YAML config file for testing."""
    config_data = {
        "data_paths": {"input": "data/input.csv", "output": "data/output.csv"},
        "parameters": {"learning_rate": 0.01},
    }

    with NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
        yaml.dump(config_data, temp_file)
        temp_path = temp_file.name

    yield temp_path  # Provide the path to the test function

    os.remove(temp_path)  # Cleanup after test


@pytest.mark.load_config
def test_load_config_valid(sample_config):
    """Test loading a valid config file."""
    config = load_config(sample_config)
    assert isinstance(config, dict)
    assert "data_paths" in config
    assert "parameters" in config
    assert os.path.isabs(config["data_paths"]["input"])  # Ensure paths are resolved


@pytest.mark.load_config
def test_load_config_missing_file():
    """Test behavior when the config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")


@pytest.mark.load_config
def test_load_config_invalid_yaml():
    """Test behavior when the YAML file is malformed."""
    with NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
        temp_file.write("invalid_yaml: [missing value")
        temp_path = temp_file.name

    with pytest.raises(ValueError):
        load_config(temp_path)

    os.remove(temp_path)


@pytest.mark.load_config
def test_load_config_non_dict():
    """Test handling of a YAML file that does not contain a dictionary."""
    with NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
        yaml.dump(["list", "instead", "of", "dict"], temp_file)
        temp_path = temp_file.name

    with pytest.raises(TypeError):
        load_config(temp_path)

    os.remove(temp_path)


@pytest.mark.load_config
def test_load_config_resolves_relative_paths(sample_config):
    """Test if relative paths in config are properly resolved to absolute paths."""
    config = load_config(sample_config)
    assert os.path.isabs(config["data_paths"]["input"])
    assert os.path.isabs(config["data_paths"]["output"])


@pytest.mark.load_config
def create_test_csv(file_path: str, num_rows: int = 100):
    """Creates a sample CSV file for testing purposes."""
    data = "id,value\n" + "\n".join([f"{i},{i*10}" for i in range(num_rows)])
    with open(file_path, "w") as f:
        f.write(data)


@pytest.mark.load_sampled_data
def test_load_entire_file(tmp_path):
    """Test loading the entire CSV file."""
    test_file = tmp_path / "test.csv"
    create_test_csv(test_file)
    df = load_sampled_data(test_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100  # Should load all rows


@pytest.mark.load_sampled_data
def test_load_sampled_data(tmp_path):
    """Test sampling functionality."""
    test_file = tmp_path / "test.csv"
    create_test_csv(test_file)
    df = load_sampled_data(test_file, sample_size=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10  # Should sample 10 rows
    assert df.index.is_monotonic_increasing  # Ensures indices are reset


@pytest.mark.load_sampled_data
def test_load_sampled_data_chunked(tmp_path):
    """Test chunked sampling functionality."""
    test_file = tmp_path / "test.csv"
    create_test_csv(test_file)
    df = load_sampled_data(test_file, sample_size=10, use_chunks=True, chunk_size=20)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10  # Should sample 10 rows
    assert df.index.is_monotonic_increasing  # Ensures indices are reset


@pytest.mark.load_sampled_data
def test_no_sample_size_provided(tmp_path):
    """Test loading the full dataset when sample_size is None."""
    test_file = tmp_path / "test.csv"
    create_test_csv(test_file)
    df = load_sampled_data(test_file, sample_size=None)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100  # Should load all rows


@pytest.mark.load_sampled_data
def test_invalid_file_path():
    """Test handling of invalid file path."""
    with pytest.raises(FileNotFoundError):
        load_sampled_data("non_existent_file.csv")


@pytest.mark.load_sampled_data
def test_invalid_sample_size(tmp_path):
    """Test handling of invalid sample size."""
    test_file = tmp_path / "test.csv"
    create_test_csv(test_file)
    with pytest.raises(ValueError):
        load_sampled_data(test_file, sample_size=-5)


@pytest.mark.load_sampled_data
def test_invalid_chunk_size(tmp_path):
    """Test handling of invalid chunk size."""
    test_file = tmp_path / "test.csv"
    create_test_csv(test_file)
    with pytest.raises(ValueError):
        load_sampled_data(test_file, sample_size=10, use_chunks=True, chunk_size=-10)


@pytest.mark.sanity_check
def test_sanity_check_passes(sample_torch_data, simple_model):
    """Test that sanity check passes when loss decreases significantly."""
    device = torch.device("cpu")
    result = sanity_check(
        simple_model,
        sample_torch_data,
        device,
        max_iters=50,
        lr=1e-2,
        loss_threshold=0.75,
    )
    assert result is True, "Sanity check should pass when loss decreases."


@pytest.mark.sanity_check
def test_sanity_check_fails(sample_torch_data, simple_model):
    """Test that sanity check fails when model parameters are frozen (no learning)."""
    device = torch.device("cpu")
    for param in simple_model.parameters():
        param.requires_grad = False  # Freeze parameters to prevent learning
    result = sanity_check(
        simple_model,
        sample_torch_data,
        device,
        max_iters=50,
        lr=1e-2,
        loss_threshold=0.5,
    )
    assert result is False, "Sanity check should fail when model does not learn."


@pytest.mark.sanity_check
def test_sanity_check_edge_case(sample_torch_data, simple_model):
    """Test edge case where max_iters is very low, which might not allow loss to drop significantly."""
    device = torch.device("cpu")
    result = sanity_check(
        simple_model,
        sample_torch_data,
        device,
        max_iters=1,
        lr=1e-2,
        loss_threshold=0.5,
    )
    assert (
        result is False
    ), "Sanity check should fail with insufficient training iterations."


@pytest.mark.create_dataloader
def test_create_dataloader_valid(sample_pandas_data):
    """Test DataLoader creation with valid data."""
    X, y = sample_pandas_data
    dataloader = create_dataloader(X, y, batch_size=2)
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 3  # Should create 3 batches of size 2, 2, and 1


@pytest.mark.create_dataloader
def test_create_dataloader_mismatched_lengths():
    """Test DataLoader raises an error when X and y have different lengths."""
    X = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})
    y = pd.Series([1])  # Mismatched length
    with pytest.raises(ValueError, match="must have the same number of samples"):
        create_dataloader(X, y)


@pytest.mark.create_dataloader
def test_create_dataloader_empty():
    """Test DataLoader raises an error when X or y is empty."""
    X = pd.DataFrame()
    y = pd.Series()
    with pytest.raises(ValueError, match="cannot be empty"):
        create_dataloader(X, y)


@pytest.mark.create_dataloader
def test_create_dataloader_tensor_shapes(sample_pandas_data):
    """Test if tensors have the correct shape in the DataLoader."""
    X, y = sample_pandas_data
    dataloader = create_dataloader(X, y, batch_size=2)
    batch_X, batch_y = next(iter(dataloader))
    assert batch_X.shape == (2, X.shape[1])
    assert batch_y.shape == (2,)
