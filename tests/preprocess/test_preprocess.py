# Add sys path to import src module
import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from preprocess import (
    load_config,
    load_dataset,
    merge_chemical_and_y,
    merge_with_transcriptomic,
    partition_data,
    preprocess_data,
    preprocess_transcriptomic_features,
    save_dataset,
)


@pytest.fixture
def config():
    return {
        "data_paths": {
            "compoundinfo_file": "tests/data/compoundinfo.csv",
            "x_file": "tests/data/x_file.tsv",
            "y_file": "tests/data/y_file.tsv",
            "output_file": "tests/data/output.csv",
        },
        "preprocess_params": {"scale_features": True},
    }


@patch("builtins.open", new_callable=mock_open, read_data="data")
@patch("yaml.safe_load", return_value={"data_paths": {}, "preprocess_params": {}})
def test_load_config(mock_safe_load, mock_open):
    config = load_config("config.yaml")
    mock_open.assert_called_once_with("config.yaml", "r")
    mock_safe_load.assert_called_once()
    assert config == {"data_paths": {}, "preprocess_params": {}}


@patch("pandas.read_csv")
def test_preprocess_data(mock_read_csv, config):
    # Mock dataframes
    compound_df = pd.DataFrame(
        {"pert_id": ["id1", "id2"], "canonical_smiles": ["C1=CC=CC=C1", "C1=CC=CC=C1"]}
    )
    x_df = pd.DataFrame({"cell_line": ["line1", "line2"], 1: [0.1, 0.2], 2: [0.3, 0.4]})
    y_df = pd.DataFrame(
        {
            "pert_mfc_id": ["id1", "id2"],
            "sig_id": ["line1", "line2"],
            "viability": [0.5, 0.6],
        }
    )

    mock_read_csv.side_effect = [compound_df, x_df, y_df]

    with patch("src.preprocess.preprocess.save_dataset") as mock_save_dataset:
        preprocess_data(config)
        assert mock_read_csv.call_count == 3
        mock_save_dataset.assert_called_once()


def test_preprocess_data_file_not_found(config):
    with patch("pandas.read_csv", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            preprocess_data(config)


def test_preprocess_data_parser_error(config):
    with patch("pandas.read_csv", side_effect=pd.errors.ParserError):
        with pytest.raises(pd.errors.ParserError):
            preprocess_data(config)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="data_paths:\n  compoundinfo_file: 'compound.csv'\n  x_file: 'x.tsv'\n  y_file: 'y.tsv'\n  output_file: 'output.csv'\npreprocess_params:\n  scale_features: True",
)
def test_load_config_file(mock_file):
    config = load_config("config.yaml")
    assert config["data_paths"]["compoundinfo_file"] == "compound.csv"
    assert config["preprocess_params"]["scale_features"] == True


@patch("pandas.read_csv")
def test_load_dataset(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df = load_dataset("test.csv")
    assert df.shape == (2, 2)


@patch("pandas.read_csv")
def test_load_dataset_file_not_found(mock_read_csv):
    mock_read_csv.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent.csv")


@patch("pandas.read_csv")
def test_load_dataset_parser_error(mock_read_csv):
    mock_read_csv.side_effect = pd.errors.ParserError
    with pytest.raises(pd.errors.ParserError):
        load_dataset("test.csv")


def test_merge_chemical_and_y():
    y_df = pd.DataFrame(
        {"pert_mfc_id": ["id1"], "sig_id": ["sig1"], "viability": [0.5]}
    )
    compound_df = pd.DataFrame(
        {"pert_id": ["id1"], "canonical_smiles": ["C1=CC=CC=C1"]}
    )
    merged_df = merge_chemical_and_y(y_df, compound_df)
    assert merged_df.shape == (1, 4)


def test_merge_with_transcriptomic():
    merged_df = pd.DataFrame({"sig_id": ["sig1"], "other_data": [1]})
    x_df = pd.DataFrame({"cell_line": ["sig1"], 1: [0.1], 2: [0.2]})
    final_df = merge_with_transcriptomic(merged_df, x_df)
    assert final_df.shape == (1, 4)


def test_preprocess_transcriptomic_features():
    final_df = pd.DataFrame({"cell_line": ["sig1"], 1: [0.1], 2: [0.2]})
    x_df = pd.DataFrame({"cell_line": ["sig1"], 1: [0.1], 2: [0.2]})
    processed_df = preprocess_transcriptomic_features(final_df, x_df, True)
    assert "cell_line" in processed_df.columns


@patch("pandas.DataFrame.to_csv")
def test_save_dataset(mock_to_csv):
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    save_dataset(df, "output.csv")
    mock_to_csv.assert_called_once_with("output.csv", index=False)


def test_partition_data():
    final_df = pd.DataFrame(
        {
            "canonical_smiles": ["C1=CC=CC=C1"],
            "viability": [0.5],
            **{i: [0.1] for i in range(1, 683)},
        }
    )
    chemical_compounds_df, viability_df, gene_expression_df = partition_data(final_df)
    assert chemical_compounds_df.shape == (1, 1)
    assert viability_df.shape == (1, 1)
    assert gene_expression_df.shape == (1, 682)
