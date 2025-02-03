import csv
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import (
    _merge_data,
    _select_genes,
    _write_df_in_chunks,
    preprocess_gene_data,
    run_tf_activity_inference,
    split_data,
)


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


@pytest.fixture
def geneinfo_df():
    """Return a dummy geneinfo DataFrame."""
    data = {
        "gene_symbol": ["GeneA", "GeneB", "GeneC", "GeneD"],
        "feature_space": ["landmark", "best inferred", "other", "landmark"],
    }
    # Use a RangeIndex that corresponds to column indices.
    df = pd.DataFrame(data)
    df.index = [0, 1, 2, 3]
    return df


@pytest.fixture
def X_df():
    """Dummy gene expression DataFrame."""
    data = {"GeneA": [1, 2], "GeneB": [3, 4]}
    return pd.DataFrame(data)


@pytest.fixture
def y_df():
    """Dummy metadata DataFrame with expected merge columns."""
    data = {
        "viability": [0.9, 0.8],
        "cell_mfc_name": ["Cell1", "Cell2"],
        "pert_dose": [10, 20],
        "other": [100, 200],
    }
    return pd.DataFrame(data)


def test_select_genes_landmark(geneinfo_df):

    selected = _select_genes(geneinfo_df, "landmark")
    # Expect only rows with feature_space "landmark"
    expected = geneinfo_df[geneinfo_df.feature_space == "landmark"]
    pd.testing.assert_frame_equal(selected, expected)


def test_select_genes_best_inferred(geneinfo_df):
    selected = _select_genes(geneinfo_df, "best inferred")
    expected = geneinfo_df[
        geneinfo_df.feature_space.isin(["landmark", "best inferred"])
    ]
    pd.testing.assert_frame_equal(selected, expected)


def test_select_genes_all(geneinfo_df):
    selected = _select_genes(geneinfo_df, "all")
    pd.testing.assert_frame_equal(selected, geneinfo_df)


def test_select_genes_invalid(geneinfo_df):
    with pytest.raises(ValueError):
        _select_genes(geneinfo_df, "invalid_space")


def test_merge_data_success(X_df, y_df):

    merged = _merge_data(X_df, y_df)
    # Check that merged contains all columns from X_df and the three merge columns
    expected = pd.concat(
        [X_df, y_df[["viability", "cell_mfc_name", "pert_dose"]]], axis=1
    )
    pd.testing.assert_frame_equal(merged, expected)


def test_merge_data_missing_column(X_df, y_df):
    # Remove one of the required merge columns
    y_df_missing = y_df.drop(columns=["pert_dose"])
    with pytest.raises(KeyError):
        _merge_data(X_df, y_df_missing)


def test_write_df_in_chunks(tmp_path):

    # Create a dummy DataFrame with 10 rows.
    df = pd.DataFrame({"col1": range(10), "col2": range(10, 20)})
    output_file = tmp_path / "output.csv"
    chunk_size = 3
    _write_df_in_chunks(df, str(output_file), chunk_size)

    # Read back the CSV and compare to original DataFrame.
    # Since to_csv writes without an index, read_csv will have a default RangeIndex.
    read_df = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(read_df, df.reset_index(drop=True))


@pytest.fixture
def dummy_rna_data(tmp_path):
    """
    Create a dummy binary file to be used as RNA data.
    Create a small 2D array with shape (num_rows, num_genes).
    """
    num_rows = 5
    num_genes = 4  # Must match geneinfo rows that we will create.
    data = np.arange(num_rows * num_genes, dtype=np.float32).reshape(
        num_rows, num_genes
    )
    # Write binary file
    rna_file = tmp_path / "rna.bin"
    data.tofile(rna_file)
    return str(rna_file), data.shape


@pytest.fixture
def dummy_geneinfo(tmp_path):
    """
    Create a dummy geneinfo CSV file.
    The index of the DataFrame will be used to select columns from RNA data.
    """
    df = pd.DataFrame(
        {
            "gene_symbol": ["GeneA", "GeneB", "GeneC", "GeneD"],
            "feature_space": ["landmark", "best inferred", "other", "landmark"],
        }
    )
    file_path = tmp_path / "geneinfo.tsv"
    df.to_csv(file_path, sep="\t", index=False)
    return str(file_path), df


@pytest.fixture
def dummy_y(tmp_path):
    """
    Create a dummy y CSV file with metadata.
    The number of rows should match the RNA data (num_rows = 5).
    """
    df = pd.DataFrame(
        {
            "viability": [0.9, 0.8, 0.85, 0.95, 0.75],
            "cell_mfc_name": ["Cell1", "Cell2", "Cell3", "Cell4", "Cell5"],
            "pert_dose": [10, 20, 15, 10, 25],
        }
    )
    file_path = tmp_path / "y.tsv"
    df.to_csv(file_path, sep="\t", index=False)
    return str(file_path), df


def test_preprocess_gene_data_invalid_shape(
    tmp_path, dummy_rna_data, dummy_geneinfo, dummy_y
):
    """
    Test the error case when RNA data has fewer columns than expected.
    We simulate this by making the dummy geneinfo have extra rows.
    """
    # Create RNA data with shape (5, 4)
    rna_file, shape = dummy_rna_data
    # Now create geneinfo with more rows than available columns
    df = pd.DataFrame(
        {
            "gene_symbol": ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"],
            "feature_space": [
                "landmark",
                "best inferred",
                "other",
                "landmark",
                "landmark",
            ],
        }
    )
    geneinfo_file = tmp_path / "geneinfo_extra.tsv"
    df.to_csv(geneinfo_file, sep="\t", index=False)

    # Use the same y file
    _, y_df = dummy_y
    y_file = tmp_path / "y.tsv"
    y_df.to_csv(y_file, sep="\t", index=False)

    preprocessed_gene_file = str(tmp_path / "preprocessed_gene.csv")
    preprocessed_landmark_file = str(tmp_path / "preprocessed_landmark.csv")
    preprocessed_best_inferred_file = str(tmp_path / "preprocessed_best_inferred.csv")

    config = {
        "data_paths": {
            "rna_file": rna_file,
            "geneinfo_file": str(geneinfo_file),
            "y_file": str(y_file),
            "preprocessed_gene_file": preprocessed_gene_file,
            "preprocessed_landmark_file": preprocessed_landmark_file,
            "preprocessed_best_inferred_file": preprocessed_best_inferred_file,
        }
    }

    # Expect a ValueError because geneinfo now has 5 rows but RNA data only has 4 columns.
    with pytest.raises(ValueError):
        preprocess_gene_data(
            config, standardize=True, feature_space="all", chunk_size=2
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
