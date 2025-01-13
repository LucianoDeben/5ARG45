import random
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
