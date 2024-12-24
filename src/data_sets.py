import random
from collections import defaultdict
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PerturbationDataset(Dataset):
    """
    A flexible dataset for pairing control and perturbation samples from HDF5 files.
    Includes:
      - Lazy on-the-fly row reads (no full in-memory load),
      - Optional LRU caching of rows for repeated access,
      - Flexible pairing strategies,
      - Optional normalization based on control data,
      - Debug-limited n_rows for smaller dev testing.

    Requirements for your HDF5:
      - "row_ids": shape=(num_rows,)
      - "row_metadata/*": columns each shape=(num_rows,)
      - "data": shape=(num_rows, num_genes)
      - Possibly chunked with compression for faster random row reads.
    """

    def __init__(
        self,
        controls_file: str,
        perturbations_file: str,
        smiles_dict: dict | None = None,
        plate_column: str = "det_plate",
        pairing: str = "random",
        max_pairs_per_plate: int | None = None,
        n_rows: int | None = None,
        normalize: bool = True,
        use_lru_cache: bool = False,
        # If you'd like more advanced normalization (like "all" or standard scaling),
        # you can add a parameter for that as well. For simplicity, we keep the old approach.
    ):
        super().__init__()

        self.controls_file = controls_file
        self.perturbations_file = perturbations_file
        self.smiles_dict = smiles_dict or {}
        self.plate_column = plate_column
        self.pairing = pairing
        self.max_pairs_per_plate = max_pairs_per_plate
        self.n_rows = n_rows
        self.normalize = normalize
        self.use_lru_cache = use_lru_cache

        # 1) Load just row metadata + row_ids (not the entire data matrix)
        (
            self.ctrl_metadata,
            self.ctrl_row_ids,
            self.ctrl_num_genes,
        ) = self._load_metadata(self.controls_file, n_rows)
        (
            self.pert_metadata,
            self.pert_row_ids,
            self.pert_num_genes,
        ) = self._load_metadata(self.perturbations_file, n_rows)

        # 2) Build control-perturbation pairs
        self._create_pairs()

        # 3) Precompute min/max if needed (from control data)
        if self.normalize:
            self._compute_minmax()

        # 4) Prepare a caching mechanism if desired
        if self.use_lru_cache:

            @lru_cache(maxsize=1024)
            def read_row_cached(h5_path, row_index):
                with h5py.File(h5_path, "r") as f:
                    return f["data"][row_index, :]

            self.read_row = read_row_cached
        else:

            def read_row_nocache(h5_path, row_index):
                with h5py.File(h5_path, "r") as f:
                    return f["data"][row_index, :]

            self.read_row = read_row_nocache

    def _load_metadata(self, h5_path, max_rows=None):
        """
        Read row_ids and row_metadata from an HDF5, plus figure out the shape of 'data'.
        Returns: (metadata_df, row_ids, num_genes)
        """
        with h5py.File(h5_path, "r") as f:
            if max_rows is not None:
                row_ids = f["row_ids"][:max_rows].astype(str)
            else:
                row_ids = f["row_ids"][:].astype(str)

            row_meta_dict = {}
            for col in f["row_metadata"].keys():
                if max_rows is not None:
                    row_meta_dict[col] = f[f"row_metadata/{col}"][:max_rows].astype(str)
                else:
                    row_meta_dict[col] = f[f"row_metadata/{col}"][:].astype(str)

            meta_df = pd.DataFrame(row_meta_dict)
            meta_df.set_index("sample_id", inplace=True)

            num_genes = f["data"].shape[1]

        return meta_df, row_ids, num_genes

    def _create_pairs(self):
        """
        Create (control, perturbation) pairs using self.pairing strategy.
        """
        ctrl_plates = self._group_by_plate(self.ctrl_metadata)
        pert_plates = self._group_by_plate(self.pert_metadata)

        common_plates = set(ctrl_plates.keys()).intersection(pert_plates.keys())
        self.pairs = []

        for plate_id in common_plates:
            c_ids = ctrl_plates[plate_id]
            p_ids = pert_plates[plate_id]

            if self.pairing == "combinatoric":
                for cid in c_ids:
                    for pid in p_ids:
                        self.pairs.append((cid, pid))

            elif self.pairing == "limited":
                if not self.max_pairs_per_plate:
                    raise ValueError(
                        "Must specify max_pairs_per_plate for 'limited' mode."
                    )
                # Each perturbation included once + random fill
                sampled_pairs = []
                for pid in p_ids:
                    cid = random.choice(c_ids)
                    sampled_pairs.append((cid, pid))

                all_pairs = [(cid, pid) for cid in c_ids for pid in p_ids]
                remain = max(0, self.max_pairs_per_plate - len(sampled_pairs))
                if remain > 0:
                    random_pairs = random.sample(all_pairs, min(remain, len(all_pairs)))
                    sampled_pairs.extend(random_pairs)

                self.pairs.extend(sampled_pairs)

            elif self.pairing == "random":
                # For each perturbation, pick a random control
                for pid in p_ids:
                    cid = random.choice(c_ids)
                    self.pairs.append((cid, pid))
            else:
                raise ValueError(f"Invalid pairing: {self.pairing}")

    def _group_by_plate(self, meta_df):
        from collections import defaultdict

        plate_to_samples = defaultdict(list)
        for sample_id, row in meta_df.iterrows():
            plate_id = row[self.plate_column]
            plate_to_samples[plate_id].append(sample_id)
        return plate_to_samples

    def _compute_minmax(self):
        """
        For normalization, compute min & max from the control data only.
        """
        with h5py.File(self.controls_file, "r") as f:
            if self.n_rows is not None:
                ctrl_data = f["data"][: self.n_rows, :]
            else:
                ctrl_data = f["data"][:, :]

            ctrl_data = ctrl_data.astype(np.float32)

        self.global_min = ctrl_data.min(axis=0)
        self.global_max = ctrl_data.max(axis=0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return a single item with "features", "labels", "smiles", "metadata".
        On-the-fly read from HDF5 or from LRU cache if enabled.
        """
        ctrl_id, pert_id = self.pairs[idx]

        # row indices in metadata
        ctrl_idx = self.ctrl_metadata.index.get_loc(ctrl_id)
        pert_idx = self.pert_metadata.index.get_loc(pert_id)

        # read row from HDF5 or cache
        ctrl_expr = self.read_row(self.controls_file, ctrl_idx)
        pert_expr = self.read_row(self.perturbations_file, pert_idx)

        if self.normalize:
            rng = self.global_max - self.global_min
            rng[rng == 0] = 1e-8
            ctrl_expr = (ctrl_expr - self.global_min) / rng
            pert_expr = (pert_expr - self.global_min) / rng

        # metadata
        ctrl_meta = self.ctrl_metadata.loc[ctrl_id].to_dict()
        pert_meta = self.pert_metadata.loc[pert_id].to_dict()

        # smiles
        smiles_str = None
        if self.smiles_dict is not None:
            compound_id = pert_meta.get("pert_id", None)
            if compound_id is not None:
                smiles_str = self.smiles_dict.get(compound_id)
            else:
                smiles_str = "N/A"

        features = torch.from_numpy(ctrl_expr.astype(np.float32))
        labels = torch.from_numpy(pert_expr.astype(np.float32))

        return {
            "features": features,
            "labels": labels,
            "smiles": smiles_str,
            "metadata": {"control_metadata": ctrl_meta, "pert_metadata": pert_meta},
        }

    ################################################################
    # Options to handle concurrency:
    ################################################################
    def worker_init_fn(self, worker_id):
        """
        If you use DataLoader(num_workers>0), you can specify:
          DataLoader(..., worker_init_fn=dataset.worker_init_fn)
        to reset the cache or any per-worker resource.
        """
        # If using an LRU cache, you might want to clear it here for each worker
        if self.use_lru_cache:
            self.read_row.cache_clear()

    ################################################################
    # Optionally, if you want to store open file references per worker:
    ################################################################
    def __del__(self):
        """
        In case references or caching need finalization, define cleanup logic here.
        """
        # If you had open file references, you'd close them here
        pass
