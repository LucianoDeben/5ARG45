import os
import zipfile

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler


def extract_zip_files(zip_dir, extract_to):
    for file_name in os.listdir(zip_dir):
        if file_name.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(zip_dir, file_name), "r") as zip_ref:
                zip_ref.extractall(extract_to)


def load_mtx_data(mtx_dir):
    barcodes = pd.read_csv(os.path.join(mtx_dir, "barcodes.tsv"), header=None, sep="\t")
    genes = pd.read_csv(os.path.join(mtx_dir, "genes.tsv"), header=None, sep="\t")
    matrix = mmread(os.path.join(mtx_dir, "matrix.mtx")).tocsc()
    classifications = pd.read_csv(os.path.join(mtx_dir, "classifications.csv"))
    return barcodes, genes, matrix, classifications


def preprocess_transcriptomics_data(matrix, classifications):
    # Filter cells based on quality
    classifications = classifications[classifications["cell_quality"] == "normal"]
    cell_indices = classifications.index
    matrix = matrix[:, cell_indices]

    # Normalize the data
    scaler = StandardScaler(with_mean=False)
    matrix = scaler.fit_transform(matrix)

    return matrix, classifications


def save_sparse_matrix_to_csv(matrix, classifications, output_file):
    # Save the classifications to the CSV file
    classifications.to_csv(output_file, index=False)

    # Append the matrix data in chunks
    chunk_size = 1000  # Adjust the chunk size as needed
    num_rows = matrix.shape[0]
    with open(output_file, "a") as f:
        for start_row in range(0, num_rows, chunk_size):
            end_row = min(start_row + chunk_size, num_rows)
            chunk = matrix[start_row:end_row].toarray()
            chunk_df = pd.DataFrame(chunk)
            chunk_df.to_csv(f, header=False, index=False)


if __name__ == "__main__":
    zip_dir = "data/test"
    extract_to = "data/processed/test"
    extract_zip_files(zip_dir, extract_to)

    for folder_name in os.listdir(extract_to):
        mtx_dir = os.path.join(extract_to, folder_name)
        if os.path.isdir(mtx_dir):
            barcodes, genes, matrix, classifications = load_mtx_data(mtx_dir)
            matrix, classifications = preprocess_transcriptomics_data(
                matrix, classifications
            )

            # Save the combined DataFrame to a CSV file
            output_file = os.path.join(mtx_dir, "final_test_dataset.csv")
            save_sparse_matrix_to_csv(matrix, classifications, output_file)
