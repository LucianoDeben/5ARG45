import os
import random
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
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
    Unified dataset for LINCS data stored in a .gctx file with optional standardization
    and gene selection.
    
    This class supports both lazy loading (for deep learning with PyTorch) and
    full in-memory loading (via to_numpy()) for classical ML models (e.g., regression,
    random forests). Additionally, it can apply standardization to the gene expression data
    and select a subset of genes based on a feature space criterion.
    
    The .gctx file is assumed to follow the LINCS convention:
      - Gene expression data at '/0/DATA/0/matrix'
      - Row metadata (experiments) under '/0/META/ROW/' (with "viability" as a column)
      - Column metadata (genes) under '/0/META/COL/' (including a 'feature_space' field)
      
    Parameters:
        h5_path (str): Path to the .gctx file.
        in_memory (bool): If True, load the entire dataset into memory.
                         If False (default), use lazy loading.
        normalize (str or None): Data standardization method. Options:
                         "min-max" for min–max scaling,
                         "z-score" for z-score standardization,
                         or None for no normalization.
        feature_space (Union[str, List[str]]): Specifies which genes to select based on the 
                         'feature_space' field in the gene metadata.
                         Allowed values are "landmark", "best inferred", and "inferred".
                         To select all genes, you may pass "all" (case-insensitive) or
                         a list containing "all" (e.g. ["all"]).
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

        # Validate normalization parameter.
        if self.normalize not in (None, "min-max", "z-score"):
            raise ValueError("normalize must be one of 'min-max', 'z-score', or None")
        
        # Validate file existence.
        if not os.path.exists(self.gctx_path):
            raise FileNotFoundError(f"The file {self.gctx_path} does not exist.")

        # --- Load Expression Data and Row Metadata ---
        if self.in_memory:
            # Full in-memory loading.
            with h5py.File(self.gctx_path, "r") as f:
                self.X = f["0/DATA/0/matrix"][:].astype(np.float32)
                viability_data = f["0/META/ROW"]["viability"][:]
                try:
                    self.y = np.array([float(x.decode("utf-8")) for x in viability_data])
                except AttributeError:
                    self.y = viability_data.astype(np.float32)
                row_ids = f["0/META/ROW"]["id"][:]
                try:
                    self.row_ids = np.array([x.decode("utf-8") for x in row_ids])
                except AttributeError:
                    self.row_ids = row_ids
        else:
            # Lazy loading: keep file open.
            self.file = h5py.File(self.gctx_path, "r")
            self.X_dataset = self.file["0/DATA/0/matrix"]
            viability_data = self.file["0/META/ROW"]["viability"][:]
            try:
                self.y = np.array([float(x.decode("utf-8")) for x in viability_data])
            except AttributeError:
                self.y = viability_data.astype(np.float32)
        
        geneinfo = self.get_col_metadata()
        selected_genes = self._select_genes(geneinfo, self.feature_space)
        # The gene metadata DataFrame index corresponds to the column order.
        selected_gene_indices = selected_genes.index.to_numpy()
        logging.info(f"Selected {len(selected_gene_indices)} genes using feature_space '{self.feature_space}'.")

        if self.in_memory:
            if self.X.shape[1] < len(selected_gene_indices):
                raise ValueError(
                    "Mismatch: The number of columns in the RNA data is less than the number of selected genes."
                )
            # Update the in-memory expression matrix to only include selected genes.
            self.X = self.X[:, selected_gene_indices]
        else:
            if self.X_dataset.shape[1] < len(selected_gene_indices):
                raise ValueError(
                    "Mismatch: The number of columns in the RNA data is less than the number of selected genes."
                )
            # For lazy loading, store the selected gene indices.
            self.selected_gene_indices = selected_gene_indices
        
        if self.normalize is not None:
            if self.in_memory:
                data = self.X  # Already subset.
            else:
                with h5py.File(self.gctx_path, "r") as f:
                    data = f["0/DATA/0/matrix"][:].astype(np.float32)
                data = data[:, self.selected_gene_indices]
            if self.normalize == "min-max":
                self.global_min = data.min(axis=0)
                self.global_max = data.max(axis=0)
            elif self.normalize == "z-score":
                self.global_mean = data.mean(axis=0)
                self.global_std = data.std(axis=0)
                self.global_std[self.global_std == 0] = 1e-8

    def get_col_metadata(self) -> pd.DataFrame:
        """
        Load the gene metadata from the /0/META/COL/ group in the .gctx file.
        Assumes that each dataset in the group corresponds to a column in the gene metadata.
        
        Returns:
            pd.DataFrame: Gene metadata with an integer index corresponding to gene order.
        """
        with h5py.File(self.gctx_path, "r") as f:
            meta_col_grp = f["0/META/COL"]
            col_dict = {}
            for key in meta_col_grp.keys():
                col_dict[key] = meta_col_grp[key][:].astype(str)
            col_df = pd.DataFrame(col_dict)
            col_df.index = np.arange(col_df.shape[0])
        return col_df

    def get_row_metadata(self) -> pd.DataFrame:
        """
        Load the row metadata from the /0/META/ROW/ group in the .gctx file.
        
        Returns:
            pd.DataFrame: Row metadata with an integer index corresponding to sample order.
        """
        with h5py.File(self.gctx_path, "r") as f:
            meta_row_grp = f["0/META/ROW"]
            row_dict = {}
            for key in meta_row_grp.keys():
                row_dict[key] = meta_row_grp[key][:].astype(str)
            row_df = pd.DataFrame(row_dict)
            row_df.index = np.arange(row_df.shape[0])
        return row_df

    def get_expression_matrix(self) -> np.ndarray:
        """
        Retrieve the full gene expression matrix. If the dataset is loaded in memory,
        return the in-memory copy; otherwise, read from the file and apply gene selection.
        
        Returns:
            np.ndarray: Gene expression matrix (float32) with selected genes.
        """
        if self.in_memory:
            return self.X
        else:
            with h5py.File(self.gctx_path, "r") as f:
                X = f["0/DATA/0/matrix"][:].astype(np.float32)
            return X[:, self.selected_gene_indices]

    @staticmethod
    def _select_genes(geneinfo: pd.DataFrame, feature_space: Union[str, List[str]]) -> pd.DataFrame:
        """
        Filter the geneinfo DataFrame based on the requested feature space(s).
        
        Parameters:
            geneinfo (pd.DataFrame): DataFrame containing gene metadata.
            feature_space (Union[str, List[str]]): Either a single feature space or a list of them.
                Allowed values are "landmark", "best inferred", and "inferred".
                If the string "all" is provided (case-insensitive) or included in a list,
                it will be converted to all three allowed values.
            
        Returns:
            pd.DataFrame: Filtered gene metadata.
        """
        allowed_values = {"landmark", "best inferred", "inferred"}
        
        # Convert feature_space into a list of values.
        if isinstance(feature_space, str):
            if feature_space.lower() == "all":
                fs = ["landmark", "best inferred", "inferred"]
            else:
                if feature_space not in allowed_values:
                    raise ValueError(f"Invalid feature_space: {feature_space}. Allowed values: {allowed_values} or 'all'")
                fs = [feature_space]
        elif isinstance(feature_space, list):
            # If "all" is in the list (case-insensitive), select all.
            if any(x.lower() == "all" for x in feature_space):
                fs = ["landmark", "best inferred", "inferred"]
            else:
                for x in feature_space:
                    if x not in allowed_values:
                        raise ValueError(f"Invalid feature_space value in list: {x}. Allowed values: {allowed_values} or 'all'")
                fs = feature_space
        else:
            raise ValueError("feature_space must be a string or a list of strings.")
        
        selected_genes = geneinfo[geneinfo.feature_space.isin(fs)]
        if selected_genes.empty:
            raise ValueError(f"No genes found for feature_space values: {fs}")
        logging.debug(f"Selected {selected_genes.shape[0]} genes for feature_space {fs}.")
        return selected_genes

    def _normalize_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply the selected normalization to the sample.
        
        Parameters:
            sample (np.ndarray): Gene expression data for one or more samples.
            
        Returns:
            np.ndarray: Normalized data.
        """
        if self.normalize == "min-max":
            rng = self.global_max - self.global_min
            rng[rng == 0] = 1e-8
            return (sample - self.global_min) / rng
        elif self.normalize == "z-score":
            return (sample - self.global_mean) / self.global_std
        else:
            return sample

    def __len__(self) -> int:
        if self.in_memory:
            return self.X.shape[0]
        else:
            return self.X_dataset.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            sample = self.X[idx, :]  # Already subset.
            label = self.y[idx]
        else:
            # Lazy loading: read only the requested row.
            sample = self.X_dataset[idx, :].astype(np.float32)
            sample = sample[self.selected_gene_indices]
            label = self.y[idx]
        
        # Apply normalization if requested.
        if self.normalize is not None:
            sample = self._normalize_sample(sample)
        
        # Convert to PyTorch tensors.
        sample_tensor = torch.tensor(sample)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return sample_tensor, label_tensor

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the entire dataset into memory as NumPy arrays.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) where X is the gene expression matrix 
                                           (float32) and y are the viability labels (float32).
        """
        if not self.in_memory:
            with h5py.File(self.gctx_path, "r") as f:
                X = f["0/DATA/0/matrix"][:].astype(np.float32)
                viability_data = f["0/META/ROW"]["viability"][:]
                try:
                    y = np.array([float(x.decode("utf-8")) for x in viability_data])
                except AttributeError:
                    y = viability_data.astype(np.float32)
            X = X[:, self.selected_gene_indices]
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
        
        
def stratified_group_split(
    dataset: LINCSDataset, 
    n_outer_splits: int = 5, 
    n_inner_splits: int = 4, 
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Performs nested stratified and group-aware splitting on the dataset.
    
    For each outer fold:
      - The outer test set is one fold.
      - The outer training set (the remaining samples) is further split into inner folds.
        These inner splits can be used for hyperparameter tuning (train/validation).
    
    Parameters:
        dataset: An instance of LINCSDataset.
        n_outer_splits: Number of outer folds (each will be the test set once).
        n_inner_splits: Number of inner splits on the outer training set.
        random_state: Seed for reproducibility.
        
    Returns:
        A list of tuples, where each tuple is:
          (outer_train_idx, outer_test_idx, inner_splits)
        inner_splits is itself a list of (inner_train_idx, inner_val_idx) tuples.
    """
    # Retrieve row metadata from the dataset.
    row_metadata: pd.DataFrame = dataset.get_row_metadata()
    if "cell_mfc_name" not in row_metadata.columns:
        raise ValueError("Row metadata must include a 'cell_mfc_name' column for splitting.")
    
    # Use the "cell_mfc_name" column as both the stratification label and grouping variable.
    groups = row_metadata["cell_mfc_name"].values
    # For stratification we use the same values.
    y = groups.copy()
    indices = np.arange(len(dataset))
    
    # Outer split: each fold will serve as the test set once.
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    
    nested_splits = []
    
    for outer_train_idx, outer_test_idx in outer_splitter.split(np.zeros(len(dataset)), y, groups):
        # Now, for the outer training set, do inner splits.
        # Get groups and y for the outer training set.
        outer_train_groups = groups[outer_train_idx]
        outer_train_y = y[outer_train_idx]
        
        inner_splitter = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        inner_splits = []
        # Note: inner splitter returns indices relative to the outer_train_idx array.
        for inner_train_rel, inner_val_rel in inner_splitter.split(np.zeros(len(outer_train_idx)), outer_train_y, outer_train_groups):
            inner_train_idx = outer_train_idx[inner_train_rel]
            inner_val_idx = outer_train_idx[inner_val_rel]
            inner_splits.append((inner_train_idx, inner_val_idx))
        
        nested_splits.append((outer_train_idx, outer_test_idx, inner_splits))
    
    return nested_splits

# Main script using nested splits with inner cross-validation.
if __name__ == "__main__":
    test_file = "../data/processed/LINCS.gctx"
    dataset = LINCSDataset(
        gctx_path=test_file,
        in_memory=True,          # or False for lazy loading
        normalize="z-score",
        feature_space="landmark"   # or a list, e.g., ["landmark", "best inferred"]
    )
    
    # Obtain the nested splits (outer and inner splits)
    nested_splits = stratified_group_split(
        dataset, n_outer_splits=5, n_inner_splits=4, random_state=42
    )
    
    # Load entire dataset in-memory.
    X_all, y_all = dataset.to_numpy()
    
    # Lists to store metrics.
    outer_metrics = []  # final test metrics per outer fold
    inner_metrics_all = []  # inner CV metrics per outer fold
    
    for fold, (outer_train_idx, outer_test_idx, inner_splits) in enumerate(nested_splits):
        logging.info(f"--- Outer Fold {fold+1} ---")
        # ---- Inner Cross-Validation on Outer Training Set ----
        inner_metrics = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
            X_inner_train = X_all[inner_train_idx]
            y_inner_train = y_all[inner_train_idx]
            X_inner_val = X_all[inner_val_idx]
            y_inner_val = y_all[inner_val_idx]
            
            # Train model on inner training set.
            model_inner = LinearRegression()
            model_inner.fit(X_inner_train, y_inner_train)
            y_inner_pred = model_inner.predict(X_inner_val)
            
            # Evaluate inner metrics.
            mse_i, mae_i, r2_i, pearson_i = evaluate_regression_metrics(y_inner_val, y_inner_pred)
            logging.info(f"Outer Fold {fold+1}, Inner Fold {inner_fold+1}: "
                         f"MSE={mse_i:.4f}, MAE={mae_i:.4f}, R²={r2_i:.4f}, Pearson={pearson_i:.4f}")
            inner_metrics.append((mse_i, mae_i, r2_i, pearson_i))
        
        inner_metrics = np.array(inner_metrics)
        inner_mean = np.mean(inner_metrics, axis=0)
        inner_std = np.std(inner_metrics, axis=0)
        logging.info(f"Outer Fold {fold+1} Inner CV Mean: {inner_mean}, Std: {inner_std}")
        inner_metrics_all.append(inner_metrics)
        
        # ---- Outer Test Evaluation ----
        X_train_outer = X_all[outer_train_idx]
        y_train_outer = y_all[outer_train_idx]
        X_test_outer = X_all[outer_test_idx]
        y_test_outer = y_all[outer_test_idx]
        
        model_outer = LinearRegression()
        model_outer.fit(X_train_outer, y_train_outer)
        y_pred_outer = model_outer.predict(X_test_outer)
        
        mse_o, mae_o, r2_o, pearson_o = evaluate_regression_metrics(y_test_outer, y_pred_outer)
        logging.info(f"Outer Fold {fold+1} Test: MSE={mse_o:.4f}, MAE={mae_o:.4f}, R²={r2_o:.4f}, Pearson={pearson_o:.4f}")
        outer_metrics.append((mse_o, mae_o, r2_o, pearson_o))
    
    outer_metrics = np.array(outer_metrics)
    overall_mean = np.mean(outer_metrics, axis=0)
    overall_std = np.std(outer_metrics, axis=0)
    metric_names = ["MSE", "MAE", "R²", "Pearson"]
    for i, name in enumerate(metric_names):
        logging.info(f"Overall {name}: mean = {overall_mean[i]:.4f}, std = {overall_std[i]:.4f}")


    
    

