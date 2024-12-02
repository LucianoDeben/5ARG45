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


# Example usage
zip_dir = "path/to/zip/files"
extract_to = "path/to/extracted/files"
extract_zip_files(zip_dir, extract_to)


def load_mtx_data(mtx_dir):
    barcodes = pd.read_csv(os.path.join(mtx_dir, "barcodes.tsv"), header=None, sep="\t")
    genes = pd.read_csv(os.path.join(mtx_dir, "genes.tsv"), header=None, sep="\t")
    matrix = mmread(os.path.join(mtx_dir, "matrix.mtx")).tocsc()
    classifications = pd.read_csv(os.path.join(mtx_dir, "classifications.csv"))
    return barcodes, genes, matrix, classifications


# Example usage
mtx_dir = "path/to/extracted/files/experiment1"
barcodes, genes, matrix, classifications = load_mtx_data(mtx_dir)


def preprocess_transcriptomics_data(matrix, classifications):
    # Filter cells based on quality
    classifications = classifications[classifications["cell_quality"] == "normal"]
    cell_indices = classifications.index
    matrix = matrix[:, cell_indices]

    # Normalize the data
    scaler = StandardScaler(with_mean=False)
    matrix = scaler.fit_transform(matrix)

    return matrix, classifications


# Example usage
matrix, classifications = preprocess_transcriptomics_data(matrix, classifications)
