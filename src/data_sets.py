import os
import random
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PerturbationDataset(Dataset):
    """
    A dataset that pairs a single control sample with a single perturbation sample
    from the same plate. Includes tokenized SMILES as part of the output.
    """

    def __init__(
        self,
        controls_file: str,
        perturbations_file: str,
        smiles_dict: dict,
        tokenizer,
        plate_column: str = "det_plate",
        normalize: str = "min-max",  # Options: "min-max", "z-score", or None
        n_rows: int | None = None,
        pairing: str = "random",
        max_pairs_per_plate: int | None = None,
        landmark_only: bool = True,
        max_smiles_length: int = 512,
    ):
        super().__init__()
        self.controls_file = controls_file
        self.perturbations_file = perturbations_file
        self.smiles_dict = smiles_dict
        self.tokenizer = tokenizer
        self.plate_column = plate_column
        self.normalize = normalize
        self.n_rows = n_rows
        self.pairing = pairing
        self.max_pairs_per_plate = max_pairs_per_plate
        self.landmark_only = landmark_only
        self.max_smiles_length = max_smiles_length

        # Load all control data + metadata
        self.ctrl_data, self.ctrl_metadata, self.ctrl_ids, self.ctrl_col_indices = (
            self._load_h5(
                controls_file, max_rows=n_rows, landmark_only=self.landmark_only
            )
        )

        # Load all perturbation data + metadata
        self.pert_data, self.pert_metadata, self.pert_ids, _ = self._load_h5(
            perturbations_file,
            max_rows=n_rows,
            landmark_only=self.landmark_only,
            col_subset=self.ctrl_col_indices,
        )

        # Precompute metadata indices for faster lookups
        self.ctrl_index_map = {
            sample_id: idx for idx, sample_id in enumerate(self.ctrl_metadata.index)
        }
        self.pert_index_map = {
            sample_id: idx for idx, sample_id in enumerate(self.pert_metadata.index)
        }

        # Precompute tokenized SMILES
        self.tokenized_smiles = {
            pert_id: (
                self.tokenizer.encode(
                    smiles,
                    add_special_tokens=True,
                    max_length=self.max_smiles_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).squeeze(0)
                if smiles != "UNKNOWN"
                else torch.tensor(
                    [self.tokenizer.pad_token_id] * self.max_smiles_length,
                    dtype=torch.long,
                )
            )
            for pert_id, smiles in self.smiles_dict.items()
        }

        # Create control-perturbation pairs based on the same plate ID
        self._create_pairs(
            pairing=self.pairing, max_pairs_per_plate=self.max_pairs_per_plate
        )

        # Compute normalization statistics if requested
        if self.normalize == "min-max":
            self._compute_minmax()
        elif self.normalize == "z-score":
            self._compute_zscore()

    def _load_h5(
        self,
        h5_path: str,
        max_rows: int | None = None,
        landmark_only: bool = False,
        col_subset: np.ndarray | None = None,
    ):
        with h5py.File(h5_path, "r") as f:
            actual_rows = f["data"].shape[0]
            rows_to_read = min(
                actual_rows, max_rows if max_rows is not None else actual_rows
            )
            row_ids = f["row_ids"][:rows_to_read].astype(str)

            row_meta_dict = {
                meta_key: f[f"row_metadata/{meta_key}"][:rows_to_read].astype(str)
                for meta_key in f["row_metadata"].keys()
            }
            metadata_df = pd.DataFrame(row_meta_dict)
            metadata_df.set_index("sample_id", inplace=True)

            if col_subset is not None:
                data_mat = f["data"][:rows_to_read, col_subset].astype(np.float32)
                return data_mat, metadata_df, row_ids, col_subset
            else:
                col_meta_dict = {
                    key: f[f"col_metadata/{key}"][:].astype(str)
                    for key in f["col_metadata"].keys()
                }
                col_meta_df = pd.DataFrame(col_meta_dict)
                if landmark_only:
                    is_landmark = col_meta_df["feature_space"] == "landmark"
                    col_indices = np.where(is_landmark)[0]
                else:
                    col_indices = np.arange(f["data"].shape[1])
                data_mat = f["data"][:rows_to_read, col_indices].astype(np.float32)
                return data_mat, metadata_df, row_ids, col_indices

    def _create_pairs(self, pairing="combinatoric", max_pairs_per_plate=None):
        ctrl_plates = self._group_by_plate(self.ctrl_metadata)
        pert_plates = self._group_by_plate(self.pert_metadata)
        common_plates = set(ctrl_plates.keys()).intersection(pert_plates.keys())
        self.pairs = []
        for plate_id in common_plates:
            c_ids = ctrl_plates[plate_id]
            p_ids = pert_plates[plate_id]
            if pairing == "combinatoric":
                self.pairs.extend([(cid, pid) for cid in c_ids for pid in p_ids])
            elif pairing == "limited":
                if max_pairs_per_plate is None:
                    raise ValueError(
                        "max_pairs_per_plate must be specified for 'limited' pairing."
                    )
                sampled_pairs = [(random.choice(c_ids), pid) for pid in p_ids]
                all_pairs = [(cid, pid) for cid in c_ids for pid in p_ids]
                remaining = max(0, max_pairs_per_plate - len(sampled_pairs))
                if remaining > 0:
                    sampled_pairs.extend(
                        random.sample(all_pairs, min(remaining, len(all_pairs)))
                    )
                self.pairs.extend(sampled_pairs)
            elif pairing == "random":
                self.pairs.extend([(random.choice(c_ids), pid) for pid in p_ids])
            else:
                raise ValueError(
                    f"Invalid pairing '{pairing}'. Choose from 'combinatoric', 'limited', or 'random'."
                )

    def _group_by_plate(self, meta_df):
        plate_to_samples = defaultdict(list)
        for sample_id, row in meta_df.iterrows():
            plate_id = row.get(self.plate_column, "UNKNOWN")
            plate_to_samples[plate_id].append(sample_id)
        return plate_to_samples

    def _compute_minmax(self):
        """Compute global min-max values for normalization."""
        self.global_min = self.ctrl_data.min(axis=0)
        self.global_max = self.ctrl_data.max(axis=0)

    def _compute_zscore(self):
        """Compute global mean and standard deviation for z-score normalization."""
        self.global_mean = self.ctrl_data.mean(axis=0)
        self.global_std = self.ctrl_data.std(axis=0)
        self.global_std[self.global_std == 0] = 1e-8  # Avoid division by zero

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ctrl_id, pert_id = self.pairs[idx]
        ctrl_idx = self.ctrl_index_map[ctrl_id]
        pert_idx = self.pert_index_map[pert_id]

        ctrl_expr = self.ctrl_data[ctrl_idx, :]
        pert_expr = self.pert_data[pert_idx, :]

        # Apply normalization
        if self.normalize == "min-max":
            rng = self.global_max - self.global_min
            rng[rng == 0] = 1e-8
            ctrl_expr = (ctrl_expr - self.global_min) / rng
            pert_expr = (pert_expr - self.global_min) / rng
        elif self.normalize == "z-score":
            ctrl_expr = (ctrl_expr - self.global_mean) / self.global_std
            pert_expr = (pert_expr - self.global_mean) / self.global_std

        ctrl_meta = self.ctrl_metadata.loc[ctrl_id].to_dict()
        pert_meta = self.pert_metadata.loc[pert_id].to_dict()

        smiles_tokens = self.tokenized_smiles.get(
            pert_meta.get("pert_id", None),
            torch.tensor(
                [self.tokenizer.pad_token_id] * self.max_smiles_length, dtype=torch.long
            ),
        )

        dosage = float(pert_meta.get("pert_dose", 0.0))
        viability = float(pert_meta.get("viability", 1.0))

        features = torch.tensor(ctrl_expr.astype(np.float32))
        labels = torch.tensor(pert_expr.astype(np.float32))

        return {
            "features": features,
            "labels": labels,
            "smiles_tokens": smiles_tokens,
            "dosage": torch.tensor(dosage, dtype=torch.float32),
            "viability": torch.tensor(viability, dtype=torch.float32),
            "metadata": {"control_metadata": ctrl_meta, "pert_metadata": pert_meta},
        }

class LINCSDataset(Dataset):
    """
    Unified dataset for LINCS data stored in a .gctx file with optional standardization and gene selection.
    
    Supports lazy loading (for deep learning with PyTorch) and full in-memory loading (for classical ML).
    Applies optional normalization ("min-max" or "z-score") and gene selection based on a feature space criterion.
    
    Assumptions:
      - Expression data is at '/0/DATA/0/matrix'
      - Row metadata (including "viability") is at '/0/META/ROW/'
      - Column metadata (including 'feature_space') is at '/0/META/COL/'
    
    Parameters:
        gctx_path (str): Path to the .gctx file.
        in_memory (bool): If True, loads all data into memory; else, uses lazy loading.
        normalize (str or None): "min-max", "z-score", or None.
        feature_space (Union[str, List[str]]): Allowed values are "landmark", "best inferred", "inferred". 
                                              Use "all" (or ["all"]) for all genes.
    """
    def __init__(
        self, 
        gctx_path: str, 
        in_memory: bool = False, 
        normalize: Optional[str] = None,
        feature_space: Union[str, List[str]] = "landmark"
    ):
        self.gctx_path = gctx_path
        self.in_memory = in_memory
        self.normalize = normalize
        self.feature_space = feature_space
        
        if self.normalize not in (None, "min-max", "z-score"):
            raise ValueError("normalize must be one of 'min-max', 'z-score', or None")
        if not os.path.exists(self.gctx_path):
            raise FileNotFoundError(f"The file {self.gctx_path} does not exist.")

        # Initialize cached metadata to None.
        self._col_metadata = None
        self._row_metadata = None

        # Load expression data and row metadata.
        if self.in_memory:
            with h5py.File(self.gctx_path, "r") as f:
                self.X = f["0/DATA/0/matrix"][:].astype(np.float32)
                viability_data = f["0/META/ROW"]["viability"][:]
                try:
                    self.y = np.array([float(x.decode("utf-8")) for x in viability_data])
                except AttributeError:
                    self.y = viability_data.astype(np.float32)
                row_ids = f["0/META/ROW"]["id"][:]
                try:
                    self._row_ids = np.array([x.decode("utf-8") for x in row_ids])
                except AttributeError:
                    self._row_ids = row_ids
        else:
            self.file = h5py.File(self.gctx_path, "r")
            self.X_dataset = self.file["0/DATA/0/matrix"]
            viability_data = self.file["0/META/ROW"]["viability"][:]
            try:
                self.y = np.array([float(x.decode("utf-8")) for x in viability_data])
            except AttributeError:
                self.y = viability_data.astype(np.float32)
        
        # Gene selection
        geneinfo = self.get_col_metadata()
        selected_genes = self._select_genes(geneinfo, self.feature_space)
        self._selected_gene_indices = selected_genes.index.to_numpy()
        logging.info(f"Selected {len(self._selected_gene_indices)} genes using feature_space '{self.feature_space}'.")

        if self.in_memory:
            if self.X.shape[1] < len(self._selected_gene_indices):
                raise ValueError("Mismatch: Expression matrix has fewer columns than selected genes.")
            self.X = self.X[:, self._selected_gene_indices]
        else:
            if self.X_dataset.shape[1] < len(self._selected_gene_indices):
                raise ValueError("Mismatch: Expression matrix has fewer columns than selected genes.")
        
        # Normalization
        if self.normalize is not None:
            if self.in_memory:
                data = self.X
            else:
                with h5py.File(self.gctx_path, "r") as f:
                    data = f["0/DATA/0/matrix"][:].astype(np.float32)
                data = data[:, self._selected_gene_indices]
            if self.normalize == "min-max":
                self.global_min = data.min(axis=0)
                self.global_max = data.max(axis=0)
            elif self.normalize == "z-score":
                self.global_mean = data.mean(axis=0)
                self.global_std = data.std(axis=0)
                self.global_std[self.global_std == 0] = 1e-8

    def get_col_metadata(self) -> pd.DataFrame:
        if self._col_metadata is None:
            with h5py.File(self.gctx_path, "r") as f:
                meta_col_grp = f["0/META/COL"]
                col_dict = {key: meta_col_grp[key][:].astype(str) for key in meta_col_grp.keys()}
                self._col_metadata = pd.DataFrame(col_dict)
                self._col_metadata.index = np.arange(self._col_metadata.shape[0])
        return self._col_metadata

    def get_row_metadata(self) -> pd.DataFrame:
        if self._row_metadata is None:
            with h5py.File(self.gctx_path, "r") as f:
                meta_row_grp = f["0/META/ROW"]
                row_dict = {key: meta_row_grp[key][:].astype(str) for key in meta_row_grp.keys()}
                self._row_metadata = pd.DataFrame(row_dict)
                self._row_metadata.index = np.arange(self._row_metadata.shape[0])
        return self._row_metadata

    @property
    def row_ids(self) -> List[str]:
        """Return the list of row IDs."""
        if self.in_memory:
            return self._row_ids.tolist() if isinstance(self._row_ids, np.ndarray) else self._row_ids
        else:
            # For lazy loading, load row metadata and extract IDs.
            return self.get_row_metadata()["id"].tolist()

    @property
    def selected_gene_indices(self) -> np.ndarray:
        """Return the selected gene indices."""
        return self._selected_gene_indices

    def get_expression_matrix(self) -> np.ndarray:
        if self.in_memory:
            return self.X
        else:
            with h5py.File(self.gctx_path, "r") as f:
                X = f["0/DATA/0/matrix"][:].astype(np.float32)
            return X[:, self._selected_gene_indices]

    @staticmethod
    def _select_genes(geneinfo: pd.DataFrame, feature_space: Union[str, List[str]]) -> pd.DataFrame:
        allowed_values = {"landmark", "best inferred", "inferred"}
        if isinstance(feature_space, str):
            fs = ["landmark", "best inferred", "inferred"] if feature_space.lower() == "all" else [feature_space]
            if fs[0] not in allowed_values:
                raise ValueError(f"Invalid feature_space: {feature_space}. Allowed: {allowed_values} or 'all'.")
        elif isinstance(feature_space, list):
            fs = ["landmark", "best inferred", "inferred"] if any(x.lower() == "all" for x in feature_space) else feature_space
            for x in fs:
                if x not in allowed_values:
                    raise ValueError(f"Invalid feature_space value: {x}. Allowed: {allowed_values} or 'all'.")
        else:
            raise ValueError("feature_space must be a string or a list of strings.")
        selected_genes = geneinfo[geneinfo.feature_space.isin(fs)]
        if selected_genes.empty:
            raise ValueError(f"No genes found for feature_space values: {fs}")
        logging.debug(f"Selected {selected_genes.shape[0]} genes for feature_space {fs}.")
        return selected_genes

    def _normalize_sample(self, sample: np.ndarray) -> np.ndarray:
        if self.normalize == "min-max":
            rng = self.global_max - self.global_min
            rng[rng == 0] = 1e-8
            return (sample - self.global_min) / rng
        elif self.normalize == "z-score":
            return (sample - self.global_mean) / self.global_std
        return sample

    def __len__(self) -> int:
        return self.X.shape[0] if self.in_memory else self.X_dataset.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            sample, label = self.X[idx, :], self.y[idx]
        else:
            sample = self.X_dataset[idx, :].astype(np.float32)[self._selected_gene_indices]
            label = self.y[idx]
        if self.normalize is not None:
            sample = self._normalize_sample(sample)
        return torch.tensor(sample), torch.tensor(label, dtype=torch.float32)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.in_memory:
            with h5py.File(self.gctx_path, "r") as f:
                X = f["0/DATA/0/matrix"][:].astype(np.float32)
                viability_data = f["0/META/ROW"]["viability"][:]
                try:
                    y = np.array([float(x.decode("utf-8")) for x in viability_data])
                except AttributeError:
                    y = viability_data.astype(np.float32)
            X = X[:, self._selected_gene_indices]
        else:
            X, y = self.X, self.y
        if self.normalize is not None:
            X = self._normalize_sample(X)
        return X, y

    def close(self) -> None:
        if not self.in_memory and hasattr(self, "file"):
            self.file.close()

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        """Enable use with the 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure resources are closed when exiting the 'with' block."""
        self.close()



    
    

