import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config_utils import load_config
from data.adapters import DatasetMetadata, LINCSAdapter

logger = logging.getLogger(__name__)


class LINCSCTRPMatcher:
    """
    Match LINCS and CTRP datasets, creating a comprehensive .gctx file.

    This class handles the complex matching process between LINCS signatures
    and CTRP drug response curves, preparing a unified dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the matcher with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Prepare file paths from config
        self.gctx_file = config["data"]["gctx_file"]
        self.geneinfo_file = config["data"]["geneinfo_file"]
        self.siginfo_file = config["data"]["siginfo_file"]

        # CTRP files from config
        self.ctrp_files = {
            "curves_post_qc": config["data"]["curves_post_qc"],
            "per_cpd_post_qc": config["data"]["per_cpd_post_qc"],
            "per_experiment": config["data"]["per_experiment"],
            "per_compound": config["data"]["per_compound"],
            "per_cell_line": config["data"]["per_cell_line"],
        }

        # Output path
        self.output_path = config["data"].get(
            "output_path",
            os.path.join(
                os.path.dirname(self.gctx_file),
                "..",
                "processed",
                "LINCS_CTRP_matched.gctx",
            ),
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Data containers
        self.geneinfo = None
        self.siginfo = None
        self.curves_df = None
        self.per_cpd_post_qc = None
        self.per_experiment = None
        self.per_compound = None
        self.per_cell_line = None

        # Matching results
        self.matched_siginfo = None
        self.matched_matrix = None
        self.matched_viabilities = None

    def load_data(self):
        """Load all necessary data files."""
        logger.info("Loading data files...")

        # Suppress DtypeWarning for pandas read_csv
        pd.options.mode.chained_assignment = None

        # Load gene info
        self.geneinfo = pd.read_csv(self.geneinfo_file, sep="\t", low_memory=False)
        self.geneinfo_df = self.geneinfo.set_index("gene_id")

        # Load signature info (24-hour timepoint, non-NaN dosage)
        self.siginfo = pd.read_csv(self.siginfo_file, sep="\t", low_memory=False)
        self.siginfo_df = self.siginfo.loc[
            (self.siginfo.pert_time == 24.0) & (~self.siginfo.pert_dose.isna())
        ].reset_index(drop=False)

        # Load CTRP files
        self.curves_df = pd.read_csv(
            self.ctrp_files["curves_post_qc"], sep="\t", low_memory=False
        )
        self.per_cpd_post_qc = pd.read_csv(
            self.ctrp_files["per_cpd_post_qc"], sep="\t", low_memory=False
        )
        self.per_experiment = pd.read_csv(
            self.ctrp_files["per_experiment"], sep="\t", low_memory=False
        )
        self.per_compound = pd.read_csv(
            self.ctrp_files["per_compound"], sep="\t", low_memory=False
        )
        self.per_cell_line = pd.read_csv(
            self.ctrp_files["per_cell_line"], sep="\t", low_memory=False
        )

        # Load GCTX file metadata
        with h5py.File(self.gctx_file, "r") as f:
            # Read metadata carefully
            self.gene_ids = np.array(f["0"]["META"]["ROW"]["id"]).astype(int)
            self.sig_ids = pd.Series(f["0"]["META"]["COL"]["id"]).astype(str)

            # Get matrix shape and dtype
            self.matrix_shape = f["0"]["DATA"]["0"]["matrix"].shape
            self.matrix_dtype = f["0"]["DATA"]["0"]["matrix"].dtype

        logger.info(f"GCTX Matrix Shape: {self.matrix_shape}")
        logger.info(f"GCTX Matrix Dtype: {self.matrix_dtype}")

    def _retrieve_matrix_rows(self, row_indices):
        """
        Retrieve specific rows from the GCTX matrix.

        Args:
            row_indices: List of row indices to retrieve

        Returns:
            Numpy array of selected matrix rows
        """
        # Check if row indices are valid
        max_rows = self.matrix_shape[0]
        if any(idx >= max_rows for idx in row_indices):
            raise ValueError(
                f"Row indices exceed matrix dimensions. Max rows: {max_rows}"
            )

        # Retrieve rows in chunks to manage memory
        chunk_size = 10000  # Adjust based on available memory
        retrieved_rows = []

        with h5py.File(self.gctx_file, "r") as f:
            matrix_dataset = f["0"]["DATA"]["0"]["matrix"]

            for i in range(0, len(row_indices), chunk_size):
                chunk_indices = row_indices[i : i + chunk_size]
                chunk = matrix_dataset[chunk_indices, :]
                retrieved_rows.append(chunk)

        return np.concatenate(retrieved_rows, axis=0)

    def _sig3upper(self, p: List[float], X: np.ndarray) -> np.ndarray:
        """
        Dose-response curve implementation from CTRP.

        Args:
            p: Curve parameters [EC50, slope, lower limit]
            X: Concentration values

        Returns:
            Transformed viability values
        """
        alpha, beta, b = p
        # Use np.clip to prevent overflow and ensure numerical stability
        exponent = np.clip(-(X - alpha) / beta, -100, 100)
        return b + (1 - b) / (1 + np.exp(exponent))

    def _match_siginfo_and_ctrp(self) -> Tuple[Dict, List[str], List[float], List[str]]:
        """
        Match LINCS signatures with CTRP drug response curves.

        Returns:
            Tuple of (index mapping, matched SMILES, matched viabilities, matched sig_ids)
        """
        # Create mappings
        master_ccl_id_mapping = self.per_experiment.set_index("experiment_id")[
            "master_ccl_id"
        ].to_dict()
        ccl_name_mapping = self.per_cell_line.set_index("master_ccl_id")[
            "ccl_name"
        ].to_dict()
        cpd_name_mapping = self.per_compound.set_index("master_cpd_id")[
            "cpd_name"
        ].to_dict()
        broad_cpd_id_mapping = self.per_compound.set_index("master_cpd_id")[
            "broad_cpd_id"
        ].to_dict()

        # Reset index to use integer indexing
        curves_df_reset = self.curves_df.reset_index(drop=True)

        # Precompute CTRP pairs
        ctrp_cell_names = [
            ccl_name_mapping.get(master_ccl_id_mapping.get(exp, ""), "")
            for exp in curves_df_reset.experiment_id
        ]
        ctrp_cpd_names = [
            broad_cpd_id_mapping.get(cpd, cpd) for cpd in curves_df_reset.master_cpd_id
        ]
        ctrp_pairs = list(zip(ctrp_cell_names, ctrp_cpd_names))

        # Storage for matched data
        all_matches = {}
        all_smiles = []
        all_viabilities = []
        all_sig_ids = []

        # Match strategies
        match_strategies = [
            (self.siginfo_df.cell_mfc_name, self.siginfo_df.pert_mfc_id),
            (self.siginfo_df.cell_mfc_name, self.siginfo_df.cmap_name),
        ]

        for cell_names, compound_names in match_strategies:
            # Create match pairs
            match_pairs = list(zip(cell_names, compound_names))

            # Find intersections using sets for efficiency
            intersections = set(match_pairs).intersection(set(ctrp_pairs))

            for pair in intersections:
                # Find indices using boolean indexing
                siginfo_mask = (cell_names == pair[0]) & (compound_names == pair[1])
                ctrp_mask = np.array(ctrp_pairs) == pair

                # Get indices of matching rows
                siginfo_indices = np.flatnonzero(siginfo_mask)
                ctrp_indices = np.flatnonzero(ctrp_mask)

                # Process each matching
                for sig_idx in siginfo_indices:
                    for local_ctrp_idx in ctrp_indices:
                        # Curve parameters (use reset index)
                        curve = curves_df_reset.iloc[local_ctrp_idx]
                        curve_params = curve[
                            ["p1_center", "p2_slope", "p4_baseline"]
                        ].tolist()

                        # Get concentration
                        subset = self.per_cpd_post_qc[
                            (self.per_cpd_post_qc.experiment_id == curve.experiment_id)
                            & (
                                self.per_cpd_post_qc.master_cpd_id
                                == curve.master_cpd_id
                            )
                        ]

                        # Compute viability
                        dose = self.siginfo_df.loc[sig_idx, "pert_dose"]
                        viability = self._sig3upper(curve_params, np.log2(dose))

                        # Store matches
                        all_matches[sig_idx] = local_ctrp_idx
                        all_smiles.append(
                            self.per_compound.loc[
                                self.per_compound.master_cpd_id == curve.master_cpd_id,
                                "cpd_smiles",
                            ].values[0]
                        )
                        all_viabilities.append(viability)
                        all_sig_ids.append(self.siginfo_df.loc[sig_idx, "sig_id"])

        return all_matches, all_smiles, all_viabilities, all_sig_ids

    def match_and_write(self):
        """
        Perform matching and write to .gctx file.
        """
        # Load data
        self.load_data()

        # Perform matching
        matches, smiles, viabilities, sig_ids = self._match_siginfo_and_ctrp()

        # Prepare output metadata
        metadata = DatasetMetadata(
            version="1.1",
            description="LINCS-CTRP Matched Dataset",
            data_source="LINCS and CTRP",
            creation_date=pd.Timestamp.now().isoformat(),
            platform="L1000",
            additional_info={
                "num_signatures": len(sig_ids),
                "num_compounds": len(set(smiles)),
            },
        )

        # Extract matrix rows
        matrix_indices = [np.where(self.sig_ids == sig_id)[0][0] for sig_id in sig_ids]
        matched_matrix = self._retrieve_matrix_rows(matrix_indices)

        # Prepare signature metadata
        matched_siginfo = self.siginfo_df.loc[list(matches.keys())].copy()
        matched_siginfo["smiles"] = smiles
        matched_siginfo["viability"] = viabilities
        matched_siginfo.set_index("sig_id", inplace=True)

        # Write to .gctx using LINCSAdapter's write method
        adapter = LINCSAdapter(
            expression_file=self.gctx_file,
            matrix_shape=matched_matrix.shape,
            row_metadata_file=self.siginfo_file,
            compound_info_file=self.ctrp_files["per_compound"],
            gene_info_file=self.geneinfo_file,
            metadata=metadata,
        )

        # Override internal data
        adapter.expression_data = matched_matrix
        adapter.row_metadata = matched_siginfo
        adapter.row_ids = list(matched_siginfo.index)
        adapter.col_metadata = pd.read_csv(self.geneinfo_file, sep="\t")
        adapter.col_ids = self.gene_ids.tolist()
        adapter.col_symbols = self.geneinfo_df.loc[self.gene_ids].gene_symbol.tolist()

        # Write to file
        adapter.write_data(self.output_path)

        logger.info(f"Matched dataset written to {self.output_path}")
        logger.info(f"Total matched signatures: {len(sig_ids)}")


def main():
    """
    Example usage of the matcher.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    config_path = os.path.join(project_root, "config.yaml")

    try:
        config = load_config(config_path)

        # Create matcher and run
        matcher = LINCSCTRPMatcher(config=config)
        matcher.match_and_write()

    except Exception as e:
        logger.error(f"Error in matching process: {e}")
        raise


if __name__ == "__main__":
    main()
