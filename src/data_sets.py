import os
import logging
import h5py
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import scale, minmax_scale, robust_scale
from tf_inference import TFInferenceRunner
import torch
from torch.utils.data import Dataset
from typing import Optional, Union, List, Tuple, Callable
import scanpy as sc
import decoupler as dc

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
        
        # # Print a sample of the pert_data
        # logging.info(f"Sample perturbation data:\n{self.pert_data[:20, :20]}")

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

# TODO: 1) Implement lazy loading for large datasets if RAM is a concern.
# TODO: 2) Implement the loading of precomputed normalization parameters.

class LINCSDataset(Dataset):
    """
    Unified dataset for LINCS data stored in a .gctx file with optional standardization,
    gene selection, and TF interference.

    Loads all data into memory using NumPy arrays (for scikit-learn compatibility)
    and returns torch.Tensors via __getitem__ for PyTorch model training.

    Features:
      - Gene selection based on a feature_space criterion.
      - Optional normalization ("min-max" or "z-score"), computed on-the-fly or with precomputed parameters.
      - TF interference via decoupler is implemented in a stateless manner.
      - Option to load only the first nrows (useful for debugging).

    Assumptions:
      - Expression data is stored at '/0/DATA/0/matrix'
      - Row metadata (including "viability") is stored at '/0/META/ROW/'
      - Column metadata (including 'feature_space') is stored at '/0/META/COL/'

    Parameters:
        gctx_path (str): Path to the .gctx file.
        normalize (str or None): "min-max", "z-score", or None.
        feature_space (Union[str, List[str]]): Allowed values are "landmark", "best inferred", "inferred". 
                                              Use "all" (or ["all"]) for all genes.
        transform (Callable, optional): Optional function to apply sample-level transformations.
        precomputed_params (dict, optional): Precomputed normalization parameters. For "min-max", provide
            {"global_min": array, "global_max": array}; for "z-score", provide {"global_mean": array, "global_std": array}.
        nrows (int, optional): If specified, only the first nrows rows are loaded (useful for debugging).
    """
    def __init__(
        self, 
        gctx_path: str, 
        normalize: Optional[str] = None,
        feature_space: Union[str, List[str]] = "landmark",
        transform: Optional[Callable] = None,
        nrows: Optional[int] = None
    ):
        self.gctx_path = gctx_path
        self.normalize = normalize
        self.feature_space = feature_space
        self.transform = transform
        self.requested_nrows = nrows

        # 1) Load the data into self.X
        with h5py.File(self.gctx_path, "r") as f:
            self.X = f["0/DATA/0/matrix"][:].astype(np.float64)
            if self.X.size == 0:
                raise ValueError("No data found in the .gctx file at '0/DATA/0/matrix'.")
            
            viability_data = f["0/META/ROW"]["viability"][:]
            self.y = self._decode_viability(viability_data)
            self.smiles = f["0/META/ROW"]["canonical_smiles"][:self._nrows].astype(str)
            row_ids = f["0/META/ROW"]["id"][:]
            self._row_ids = self._decode_ids(row_ids)

        # 2) Possibly subset rows if nrows is specified
        full_nrows = self.X.shape[0]
        self._nrows = min(self.requested_nrows, full_nrows) if self.requested_nrows is not None else full_nrows
        self.X = self.X[:self._nrows, :]
        self.y = self.y[:self._nrows]
        self._row_ids = self._row_ids[:self._nrows]
        
        # 3) Now apply scikit-learn normalization
        if self.normalize is not None:
            if self.normalize == "z-score":
                self.X = scale(self.X, copy=True)
            elif self.normalize == "min-max":
                self.X = minmax_scale(self.X, copy=True)  

        # Load column metadata and select genes.
        geneinfo = self.get_col_metadata()
        selected_genes = self._select_genes(geneinfo, self.feature_space)
        self._selected_gene_indices = selected_genes.index.to_numpy()
        logging.info(f"Selected {len(self._selected_gene_indices)} genes using feature_space '{self.feature_space}'.")
        total_genes = self.X.shape[1]
        if total_genes < len(self._selected_gene_indices):
            raise ValueError("Mismatch: Expression matrix has fewer columns than selected genes.")
        self.X = self.X[:, self._selected_gene_indices]

    def _decode_viability(self, viability_data) -> np.ndarray:
        try:
            return np.array([float(x.decode("utf-8")) for x in viability_data])
        except AttributeError:
            return viability_data.astype(np.float32)

    def _decode_ids(self, ids_data) -> List[str]:
        try:
            return [x.decode("utf-8") for x in ids_data]
        except AttributeError:
            return ids_data.tolist()

    def get_col_metadata(self) -> pd.DataFrame:
        if not hasattr(self, "_col_metadata") or self._col_metadata is None:
            with h5py.File(self.gctx_path, "r") as f:
                meta_col_grp = f["0/META/COL"]
                col_dict = {key: meta_col_grp[key][:].astype(str) for key in meta_col_grp.keys()}
                self._col_metadata = pd.DataFrame(col_dict)
                self._col_metadata.index = np.arange(self._col_metadata.shape[0])
        return self._col_metadata
    
    def get_row_metadata(self) -> pd.DataFrame:
        if not hasattr(self, "_row_metadata") or self._row_metadata is None:
            with h5py.File(self.gctx_path, "r") as f:
                meta_row_grp = f["0/META/ROW"]
                row_dict = {key: meta_row_grp[key][:].astype(str) for key in meta_row_grp.keys()}
                self._row_metadata = pd.DataFrame(row_dict)
                self._row_metadata.index = np.arange(self._row_metadata.shape[0])
        return self._row_metadata

    @property
    def row_ids(self) -> List[str]:
        return self._row_ids if isinstance(self._row_ids, list) else self._row_ids.tolist()

    @property
    def selected_gene_indices(self) -> np.ndarray:
        return self._selected_gene_indices

    def get_expression_matrix(self) -> np.ndarray:
        return self.X

    def infer_tf_activities(self, net: pd.DataFrame, runner: TFInferenceRunner):
        X = self.get_expression_matrix()
        row_ids = self.row_ids
        gene_ids = self.get_col_metadata().iloc[self.selected_gene_indices]["id"].tolist()
        return runner.run(X, net, row_ids, gene_ids)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X.copy(), self.y.copy()

    def get_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a tuple of (row_metadata, col_metadata) as pandas DataFrames.
        """
        return self.get_row_metadata(), self.get_col_metadata()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n_samples={self._nrows}, "
                f"n_genes={self.X.shape[1]}, normalize={self.normalize}, "
                f"feature_space={self.feature_space})")
    
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

    def __len__(self) -> int:
        return self._nrows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.X[idx, :]
        label = self.y[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return torch.tensor(sample), torch.tensor(label, dtype=torch.float32)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


        
        
        
        






    
    

