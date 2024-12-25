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
        normalize: bool = True,
        n_rows: int | None = None,
        pairing: str = "random",
        max_pairs_per_plate: int | None = None,
        landmark_only: bool = True,
        max_smiles_length: int = 512,
    ):
        """
        Args:
            controls_file: Path to HDF5 file containing control data and row_metadata.
            perturbations_file: Path to HDF5 file containing perturbation data and row_metadata.
            smiles_dict (dict): Dictionary mapping pert_id -> SMILES string.
            tokenizer: Tokenizer instance for encoding SMILES strings.
            plate_column (str): Name of the column in row_metadata that indicates the plate ID.
            normalize (bool): Whether to apply global min-max normalization based on controls only.
            n_rows (int): (Optional) For debugging. Limit the dataset to the first n_rows in each file.
            pairing (str): Pairing strategy: 'random', 'combinatoric', or 'limited'.
            max_pairs_per_plate (int): If pairing='limited', the maximum number of pairs to create per plate.
            landmark_only (bool): If True, only keep columns (genes) where col_metadata['feature_space']=='landmark'.
            max_smiles_length (int): Maximum length for SMILES token sequences.
        """
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

        # 1) Load all control data + row metadata into memory
        (
            self.ctrl_data,
            self.ctrl_metadata,
            self.ctrl_ids,
            self.ctrl_col_indices,
        ) = self._load_h5(
            controls_file, max_rows=n_rows, landmark_only=self.landmark_only
        )

        # 2) Load all perturbation data + row metadata into memory
        (
            self.pert_data,
            self.pert_metadata,
            self.pert_ids,
            _,
        ) = self._load_h5(
            perturbations_file,
            max_rows=n_rows,
            landmark_only=self.landmark_only,
            col_subset=self.ctrl_col_indices,
        )

        # 3) Create control-perturbation pairs based on the same plate ID
        self._create_pairs(
            pairing=self.pairing, max_pairs_per_plate=self.max_pairs_per_plate
        )

        # 4) If requested, compute min-max across entire dataset and store for scaling
        if self.normalize:
            self._compute_minmax()

    def _load_h5(
        self,
        h5_path: str,
        max_rows: int | None = None,
        landmark_only: bool = False,
        col_subset: np.ndarray | None = None,
    ):
        """
        Read an HDF5 file fully into memory (or the first `max_rows` rows).
        If `landmark_only=True`, subset columns to those in which
        col_metadata['feature_space']=='landmark'.
        If `col_subset` is provided, we skip reading col_metadata and just use that subset.

        Returns:
            data_mat: np.ndarray of shape (rows_to_read, selected_gene_count)
            metadata_df: pd.DataFrame for row_metadata (length = rows_to_read)
            row_ids: np.ndarray[str] of shape (rows_to_read,)
            col_indices: np.ndarray of the selected column indices in the full matrix
                        (useful to ensure control and perturbation have the same columns)
        """
        with h5py.File(h5_path, "r") as f:
            # 1) Figure out how many rows exist
            actual_rows = f["data"].shape[0]

            # 2) Determine how many rows to read
            rows_to_read = min(
                actual_rows, max_rows if max_rows is not None else actual_rows
            )

            # 3) Read row_ids and row_metadata for those rows
            row_ids = f["row_ids"][:rows_to_read].astype(str)

            # Build a DataFrame from row_metadata
            row_meta_dict = {}
            for meta_key in f["row_metadata"].keys():
                full_column = f[f"row_metadata/{meta_key}"][:rows_to_read].astype(str)
                row_meta_dict[meta_key] = full_column

            metadata_df = pd.DataFrame(row_meta_dict)
            metadata_df.set_index("sample_id", inplace=True)

            # 4) If we have a pre-defined col_subset, skip col_metadata reading
            if col_subset is not None:
                # Subset columns to col_subset
                data_mat = f["data"][:rows_to_read, col_subset]
                data_mat = data_mat.astype(np.float32)
                return data_mat, metadata_df, row_ids, col_subset
            else:
                # We might need to read col_metadata to find landmark columns or else use all
                num_total_cols = f["data"].shape[1]

                # Read col_metadata
                col_meta_dict = {}
                for col2 in f["col_metadata"].keys():
                    col_meta_dict[col2] = f[f"col_metadata/{col2}"][:].astype(str)

                col_meta_df = pd.DataFrame(col_meta_dict)
                # col_meta_df has shape (#genes, #metadata_columns), e.g.:
                # gene_id, gene_symbol, ensembl_id, gene_title, gene_type, src, feature_space

                if landmark_only:
                    # find columns where feature_space == 'landmark'
                    is_landmark = col_meta_df["feature_space"] == "landmark"
                    col_indices = np.where(is_landmark)[0]
                else:
                    col_indices = np.arange(num_total_cols)

                # Subset data
                data_mat = f["data"][:rows_to_read, col_indices].astype(np.float32)

                return data_mat, metadata_df, row_ids, col_indices

    def _create_pairs(self, pairing="combinatoric", max_pairs_per_plate=None):
        """
        Create control-perturbation pairs based on the specified pairing.
        """
        ctrl_plates = self._group_by_plate(self.ctrl_metadata)
        pert_plates = self._group_by_plate(self.pert_metadata)

        common_plates = set(ctrl_plates.keys()).intersection(pert_plates.keys())
        self.pairs = []

        for plate_id in common_plates:
            c_ids = ctrl_plates[plate_id]  # List of control sample IDs
            p_ids = pert_plates[plate_id]  # List of perturbation sample IDs

            if pairing == "combinatoric":
                self.pairs.extend([(cid, pid) for cid in c_ids for pid in p_ids])

            elif pairing == "limited":
                if max_pairs_per_plate is None:
                    raise ValueError(
                        "max_pairs_per_plate must be specified for 'limited' pairing."
                    )
                # Ensure every perturbation is included at least once
                sampled_pairs = []
                for pid in p_ids:
                    cid = random.choice(c_ids)
                    sampled_pairs.append((cid, pid))

                # Fill remaining slots up to max_pairs_per_plate
                all_pairs = [(cid, pid) for cid in c_ids for pid in p_ids]
                remain = max(0, max_pairs_per_plate - len(sampled_pairs))
                if remain > 0:
                    random_pairs = random.sample(all_pairs, min(remain, len(all_pairs)))
                    sampled_pairs.extend(random_pairs)

                self.pairs.extend(sampled_pairs)

            elif pairing == "random":
                # For each perturbation, randomly sample a control
                for pid in p_ids:
                    cid = random.choice(c_ids)
                    self.pairs.append((cid, pid))

            else:
                raise ValueError(
                    f"Invalid pairing '{pairing}'. Choose from 'combinatoric', 'limited', or 'random'."
                )

    def _group_by_plate(self, meta_df):
        plate_to_samples = defaultdict(list)
        for sample_id, row in meta_df.iterrows():
            plate_id = row[self.plate_column]
            plate_to_samples[plate_id].append(sample_id)
        return plate_to_samples

    def _compute_minmax(self):
        """
        For normalization, we compute min and max from the control data only,
        then apply to both controls and perturbations in __getitem__.
        """
        self.global_min = self.ctrl_data.min(axis=0)
        self.global_max = self.ctrl_data.max(axis=0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return a single item with "features", "labels", "smiles_tokens", "metadata".
        """
        ctrl_id, pert_id = self.pairs[idx]

        # Retrieve indices from metadata
        ctrl_idx = self.ctrl_metadata.index.get_loc(ctrl_id)
        pert_idx = self.pert_metadata.index.get_loc(pert_id)

        # Read from memory
        ctrl_expr = self.ctrl_data[ctrl_idx, :]
        pert_expr = self.pert_data[pert_idx, :]

        # Optional min-max normalization
        if self.normalize:
            rng = self.global_max - self.global_min
            rng[rng == 0] = 1e-8
            ctrl_expr = (ctrl_expr - self.global_min) / rng
            pert_expr = (pert_expr - self.global_min) / rng

        # Fetch metadata
        ctrl_meta = self.ctrl_metadata.loc[ctrl_id].to_dict()
        pert_meta = self.pert_metadata.loc[pert_id].to_dict()

        # Retrieve and tokenize SMILES string
        smiles_str = self.smiles_dict.get(pert_meta.get("pert_id", None), "UNKNOWN")
        if smiles_str != "UNKNOWN":
            smiles_tokens = self.tokenizer.encode(
                smiles_str,
                add_special_tokens=True,
                max_length=self.max_smiles_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).squeeze(0)
        else:
            smiles_tokens = torch.tensor(
                [self.tokenizer.pad_token_id] * self.max_smiles_length, dtype=torch.long
            )

        # Convert to PyTorch tensors
        features = torch.tensor(ctrl_expr.astype(np.float32))
        labels = torch.tensor(pert_expr.astype(np.float32))

        return {
            "features": features,
            "labels": labels,
            "smiles_tokens": smiles_tokens,
            "metadata": {"control_metadata": ctrl_meta, "pert_metadata": pert_meta},
        }
