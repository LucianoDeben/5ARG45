# Add src to the Python path
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocess.preprocess import load_dataset


def test_load_dataset_success(tmp_path):
    # Create a temporary CSV file
    data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(data)

    # Load the dataset
    df = load_dataset(file_path)

    # Check the DataFrame content
    assert not df.empty
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == 1
    assert df.iloc[0]["col2"] == 2


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file.csv")


def test_load_dataset_parser_error(tmp_path):
    # Create a temporary file with incorrect format
    data = "col1;col2\n1;2\n3;4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(data)

    with pytest.raises(pd.errors.ParserError):
        load_dataset(file_path, delimiter=",")
