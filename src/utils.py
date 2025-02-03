import logging
import os
from typing import Any, Dict, Optional, Union

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a YAML configuration file and resolve relative paths.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary with resolved paths, or None if an error occurs.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    if not isinstance(config, dict):
        raise TypeError(
            f"Invalid configuration format in {config_path}. Expected a dictionary."
        )

    # Resolve relative paths
    config_dir = os.path.dirname(config_path)
    if "data_paths" in config and isinstance(config["data_paths"], dict):
        for key, value in config["data_paths"].items():
            if isinstance(value, str):  # Ensure only string paths are modified
                config["data_paths"][key] = os.path.abspath(
                    os.path.join(config_dir, value)
                )

    return config


def load_sampled_data(
    file_path: str,
    sample_size: Optional[int] = None,
    use_chunks: bool = False,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a dataset from a CSV file with optional sampling and chunked loading.

    Parameters:
    - file_path (str): Path to the CSV file.
    - sample_size (Optional[int]): Number of rows to sample. If None, loads the entire file.
    - use_chunks (bool): Whether to use chunked loading for large files.
    - chunk_size (Optional[int]): Size of chunks when using chunked loading.

    Returns:
    - pd.DataFrame: The loaded dataset, potentially sampled, with reset indices if sampled.
    """
    if use_chunks:
        # Load dataset in chunks
        chunks = []

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if sample_size is not None and len(chunks) * chunk_size >= sample_size:
                break
            if sample_size is not None:
                sampled_chunk = chunk.sample(
                    min(sample_size - len(chunks) * chunk_size, len(chunk)),
                    random_state=42,
                )
                chunks.append(sampled_chunk)
            else:
                chunks.append(chunk)

        sampled_data = pd.concat(chunks, axis=0, ignore_index=True)
    else:
        if sample_size is None:
            # Load entire dataset if no sample size is provided
            sampled_data = pd.read_csv(file_path)
        else:
            # Load entire file up to sample_size and then sample from it
            full_data = pd.read_csv(file_path, nrows=sample_size)
            sampled_data = full_data.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )

    return sampled_data


def sanity_check(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    max_iters: int = 100,
    lr: float = 1e-3,
    loss_threshold: float = 0.1,
) -> bool:
    """
    Perform a sanity check by training the model for a few iterations and checking if the loss decreases.

    Args:
        model (nn.Module): The PyTorch model to check.
        loader (torch.utils.data.DataLoader): DataLoader for loading a batch of data.
        device (Union[str, torch.device]): The device to run the model on.
        max_iters (int): Number of training iterations for the sanity check.
        lr (float): Learning rate for the optimizer.
        loss_threshold (float): Fraction of initial loss to compare final loss against.

    Returns:
        bool: True if the loss reduces significantly, False otherwise.
    """
    model.train()
    X, y = next(iter(loader))
    X, y = X.to(device), y.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    initial_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)

        if any(param.requires_grad for param in model.parameters()):
            loss.backward()
        else:
            logging.warning("Model parameters are frozen. Skipping backpropagation.")

        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if loss.item() < loss_threshold * initial_loss:
            logging.info("Sanity check passed.")
            return True

    logging.info("Sanity check failed.")
    return False


def create_dataloader(
    X: Union[pd.DataFrame, pd.Series],
    y: pd.Series,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from input features and labels.

    Args:
        X (Union[pd.DataFrame, pd.Series]): Feature dataset with samples as rows.
        y (pd.Series): Target variable.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader with a TensorDataset.

    Raises:
        ValueError: If X or y are empty or have mismatched lengths.
    """
    if X.empty or y.empty:
        raise ValueError("Input features and labels cannot be empty.")

    if len(X) != len(y):
        raise ValueError(
            "Feature matrix X and target variable y must have the same number of samples."
        )

    # Convert pandas DataFrame/Series to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_ontology(file_name, gene2id_mapping):
    """
    Load the ontology from a file and construct a directed graph with gene annotations.

    The file is expected to have lines formatted with three tokens. If the third token is "default",
    then the line defines an edge (parent->child) in the ontology graph. Otherwise, the line defines
    a gene annotation: the first token is the GO term and the second token is the gene symbol.
    Only genes present in `gene2id_mapping` will be considered.

    For each term in the ontology, this function computes the total number of genes that are either
    directly annotated to that term or inherited from its descendant terms. Terms with no genes are
    logged and skipped.

    Additionally, the function checks that the ontology has a single root and that the graph is
    connected.

    Args:
        file_name (str): Path to the ontology file.
        gene2id_mapping (dict): Mapping from gene symbols to gene IDs.

    Returns:
        tuple: A tuple (dG, root, term_size_map, term_direct_gene_map) where:
            - dG is the filtered directed graph (nx.DiGraph) representing the ontology.
            - root is the single root node of the ontology.
            - term_size_map is a dict mapping each term to the number of genes (after propagation)
              associated with it.
            - term_direct_gene_map is a dict mapping each term to the set of gene IDs directly annotated.

    Raises:
        ValueError: If the ontology has more than one root or is not connected.
    """
    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}
    gene_set = set()

    # Read and parse the ontology file
    with open(file_name, "r") as file_handle:
        for line in file_handle:
            tokens = line.rstrip().split()
            if len(tokens) < 3:
                continue  # Skip malformed lines
            # If token[2] is "default", this is an edge definition
            if tokens[2] == "default":
                dG.add_edge(tokens[0], tokens[1])
            else:
                # Otherwise, this is a gene annotation
                gene_symbol = tokens[1]
                if gene_symbol not in gene2id_mapping:
                    continue
                term = tokens[0]
                if term not in term_direct_gene_map:
                    term_direct_gene_map[term] = set()
                term_direct_gene_map[term].add(gene2id_mapping[gene_symbol])
                gene_set.add(gene_symbol)

    logging.info("There are %d genes in the ontology annotations.", len(gene_set))

    # Create a list to store terms that will be removed due to having no gene annotations.
    terms_to_remove = []

    # Evaluate each term in the graph
    for term in list(dG.nodes()):
        # Start with genes directly annotated to the term (if any)
        term_gene_set = set()
        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term].copy()

        # Propagate genes from descendant terms
        descendants = nxadag.descendants(dG, term)
        for child in descendants:
            if child in term_direct_gene_map:
                term_gene_set |= term_direct_gene_map[child]

        if len(term_gene_set) == 0:
            logging.warning("Term %s has no genes and will be removed.", term)
            terms_to_remove.append(term)
        else:
            term_size_map[term] = len(term_gene_set)

    # Remove empty terms from the graph and mappings
    for term in terms_to_remove:
        if term in dG:
            dG.remove_node(term)
        if term in term_direct_gene_map:
            del term_direct_gene_map[term]
        if term in term_size_map:
            del term_size_map[term]

    # Identify roots (nodes with in_degree==0)
    leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]
    if not leaves:
        raise ValueError("No root found in the ontology.")
    logging.info("There are %d roots.", len(leaves))
    if len(leaves) > 1:
        raise ValueError(
            "Multiple roots detected in ontology. Please ensure only one root exists."
        )

    root = leaves[0]

    # Check connectivity
    undirected_G = dG.to_undirected()
    connected_components = list(nxacc.connected_components(undirected_G))
    logging.info(
        "There are %d connected components in the ontology.", len(connected_components)
    )
    if len(connected_components) > 1:
        raise ValueError(
            "Ontology graph is not connected. Please connect the components."
        )

    logging.info("Ontology loaded successfully with %d terms.", len(dG.nodes()))
    return dG, root, term_size_map, term_direct_gene_map


def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()

    return mapping


def filter_dataset_columns(df, gene_mapping):
    """
    Filter the DataFrame columns so that only gene expression columns present
    in the gene_mapping are retained, along with the last three columns which
    are assumed to be 'viability', 'cell_mfc_name', and 'pert_dose'.

    Args:
        df (pd.DataFrame): The input DataFrame containing gene expression data and extra columns.
        gene_mapping (dict): A dictionary mapping gene symbols to IDs.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if df.shape[1] < 3:
        raise ValueError(
            "DataFrame does not have at least three extra columns to preserve."
        )
    gene_cols = df.columns[:-3]  # All except the last three columns
    extra_cols = df.columns[-3:]  # The last three columns
    filtered_gene_cols = [col for col in gene_cols if col in gene_mapping]
    new_columns = filtered_gene_cols + list(extra_cols)
    filtered_df = df[new_columns].copy()
    return filtered_df
