from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PertubationDataset(Dataset):
    """
    A dataset that pairs a single control sample with a single perturbation sample
    from the same plate. Data is fully loaded into memory for performance.

    If a plate has multiple controls and multiple perturbations, we enumerate all
    possible pairs (control_i, pert_j) in that plate. This ensures each __getitem__
    only does in-memory lookups.
    """

    def __init__(
        self,
        controls_file: str,
        perturbations_file: str,
        smiles_dict: dict,
        plate_column: str = "det_plate",
        normalize: bool = False,
        n_rows: int = None,
    ):
        """
        Args:
            controls_file: Path to HDF5 file containing control data and row_metadata.
            perturbations_file: Path to HDF5 file containing perturbation data and row_metadata.
            plate_column: Name of the column in row_metadata that indicates the plate ID.
            normalize: Whether to apply global min-max normalization across controls+perts.
            n_rows: (Optional) For debugging. Limit the dataset to the first n_rows in each file.
        """
        super().__init__()
        self.controls_file = controls_file
        self.perturbations_file = perturbations_file
        self.smiles_dict = smiles_dict
        self.plate_column = plate_column
        self.normalize = normalize
        self.n_rows = n_rows

        # 1) Load all control data + row metadata into memory
        self.ctrl_data, self.ctrl_metadata, self.ctrl_ids = self._load_h5(
            controls_file, max_rows=n_rows
        )

        # 2) Load all perturbation data + row metadata into memory
        self.pert_data, self.pert_metadata, self.pert_ids = self._load_h5(
            perturbations_file, max_rows=n_rows
        )

        # 3) Group sample IDs by plate for both controls and perturbations
        ctrl_plates = self._group_by_plate(self.ctrl_metadata)
        pert_plates = self._group_by_plate(self.pert_metadata)

        # 4) Enumerate all valid (control, perturbation) pairs from the same plate
        self.pairs = []
        common_plates = set(ctrl_plates.keys()).intersection(pert_plates.keys())
        for plate_id in common_plates:
            c_ids = ctrl_plates[plate_id]  # list of control sample IDs
            p_ids = pert_plates[plate_id]  # list of perturbed sample IDs
            for cid in c_ids:
                for pid in p_ids:
                    self.pairs.append((cid, pid))

        # 5) If requested, compute min-max across entire dataset and store for scaling
        if self.normalize:
            self._compute_global_minmax()

        # Done! We have a big self.pairs list of (control_id, pert_id).
        # We'll retrieve data from self.ctrl_data / self.pert_data in __getitem__.

    def _load_h5(self, h5_path, max_rows=None):
        """
        Read an HDF5 file fully into memory (or the first max_rows rows).
        Returns:
          data: A NumPy array of shape (N, G) => rows x genes
          metadata: A pandas DataFrame with row_metadata
          row_ids: The sample IDs for each row in 'data'
        """
        with h5py.File(h5_path, "r") as f:
            # If max_rows is specified, limit reading. Otherwise read all.
            if max_rows is not None:
                data_mat = f["data"][:max_rows, :]
                row_ids = f["row_ids"][:max_rows].astype(str)
            else:
                data_mat = f["data"][:, :]
                row_ids = f["row_ids"][:].astype(str)

            # Build a DataFrame from row_metadata
            row_meta_dict = {}
            for col in f["row_metadata"].keys():
                # Also slice if max_rows is used
                if max_rows is not None:
                    row_meta_dict[col] = f[f"row_metadata/{col}"][:max_rows].astype(str)
                else:
                    row_meta_dict[col] = f[f"row_metadata/{col}"][:].astype(str)

            metadata_df = pd.DataFrame(row_meta_dict)
            # We'll assume there's a "sample_id" column, or rename the index to match
            metadata_df.set_index("sample_id", inplace=True)

        data_mat = data_mat.astype(np.float32)
        return data_mat, metadata_df, row_ids

    def _group_by_plate(self, meta_df):
        """
        Group sample IDs by plate. Returns a dict:
          plate_id -> [list of sample IDs in that plate]
        """
        plate_to_samples = defaultdict(list)
        for sample_id, row in meta_df.iterrows():
            plate_id = row[self.plate_column]
            plate_to_samples[plate_id].append(sample_id)
        return plate_to_samples

    def _compute_global_minmax(self):
        """
        Compute min and max across all genes in both ctrl_data and pert_data.
        We'll use these for min-max scaling in __getitem__.
        """
        all_data = np.vstack([self.ctrl_data, self.pert_data])
        self.global_min = all_data.min(axis=0)
        self.global_max = all_data.max(axis=0)

    def __len__(self):
        # Number of (control, perturbation) pairs
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return a single (control, perturbation) pair as a dictionary
        with 'features', 'labels', 'metadata', and 'smiles'.
        """
        ctrl_id, pert_id = self.pairs[idx]

        # Lookup row index for each ID
        ctrl_idx = np.where(self.ctrl_ids == ctrl_id)[0][0]
        pert_idx = np.where(self.pert_ids == pert_id)[0][0]

        # Retrieve data from memory
        ctrl_expr = self.ctrl_data[ctrl_idx, :]
        pert_expr = self.pert_data[pert_idx, :]

        # Optional min-max normalization
        if self.normalize:
            rng = self.global_max - self.global_min
            # Avoid /0 if there's a gene with constant expression
            rng[rng == 0] = 1e-8
            ctrl_expr = (ctrl_expr - self.global_min) / rng
            pert_expr = (pert_expr - self.global_min) / rng

        # Retrieve metadata for control and perturbation
        ctrl_meta = self.ctrl_metadata.loc[ctrl_id].to_dict()
        pert_meta = self.pert_metadata.loc[pert_id].to_dict()

        # Look up SMILES string for this perturbation (if 'pert_id' key exists in the metadata)
        smiles_str = None
        if self.smiles_dict is not None:
            # e.g., pert_meta["pert_id"] might be the unique compound ID
            # Make sure your metadata actually has "pert_id" in it
            compound_id = pert_meta.get("pert_id", None)
            if compound_id is not None:
                smiles_str = self.smiles_dict.get(compound_id, None)

        # Convert expressions to PyTorch tensors
        features = torch.from_numpy(ctrl_expr)
        labels = torch.from_numpy(pert_expr)

        # Return a dictionary containing all relevant info
        return {
            "features": features,  # control expression
            "labels": labels,  # perturbation expression
            "smiles": smiles_str,  # retrieved SMILES string (if found)
            "metadata": {"control_metadata": ctrl_meta, "pert_metadata": pert_meta},
        }
