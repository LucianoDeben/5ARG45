import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5py
import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from ..config.config_utils import load_config, merge_configs, parse_cli_overrides
from ..config.constants import CURRENT_SCHEMA_VERSION, REQUIRED_CONFIG_SECTIONS
from ..data.adapters import DatasetMetadata, LINCSAdapter


class LINCSCTRPMatcher:
    """
    Advanced matcher for LINCS and CTRP datasets with robust configuration handling.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the matcher with validated configuration.

        Args:
            config: Validated configuration dictionary
            logger: Optional custom logger
        """
        # Validate configuration sections
        self._validate_config_sections(config)

        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Setup data paths from configuration
        self.paths = self._setup_paths()

        # Data containers
        self._initialize_data_containers()

        # Output path
        self.output_path = self.config["data"].get(
            "output_path",
            os.path.join(
                os.path.dirname(self.paths["gctx_file"]),
                "..",
                "processed",
                "LINCS_CTRP_matched.gctx",
            ),
        )

    def _validate_config_sections(self, config: Dict[str, Any]):
        """
        Validate that all required configuration sections are present.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If any required section is missing
        """
        missing_sections = REQUIRED_CONFIG_SECTIONS - set(config.keys())
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")

    def _setup_paths(self) -> Dict[str, str]:
        """
        Extract and validate file paths from configuration.

        Returns:
            Dictionary of validated file paths
        """
        data_config = self.config["data"]
        required_files = [
            "gctx_file",
            "geneinfo_file",
            "siginfo_file",
            "curves_post_qc",
            "per_cpd_post_qc",
            "per_experiment",
            "per_compound",
            "per_cell_line",
        ]

        paths = {}
        for file_key in required_files:
            path = data_config.get(file_key)
            if not path:
                raise ValueError(f"Missing path for {file_key}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            paths[file_key] = path

        return paths

    def _initialize_data_containers(self):
        """Initialize data containers with None."""
        data_containers = [
            "geneinfo",
            "siginfo",
            "curves_df",
            "per_cpd_post_qc",
            "per_experiment",
            "per_compound",
            "per_cell_line",
        ]

        for container in data_containers:
            setattr(self, container, None)

    def load_data(self):
        """
        Load all necessary data files with robust error handling.
        """
        self.logger.info("Loading data files...")

        try:
            # Suppress DtypeWarning for pandas read_csv
            pd.options.mode.chained_assignment = None

            # Load gene info
            self.geneinfo = pd.read_csv(
                self.paths["geneinfo_file"], sep="\t", low_memory=False
            )
            self.geneinfo_df = self.geneinfo.set_index("gene_id")

            # Load signature info (24-hour timepoint, non-NaN dosage)
            self.siginfo = pd.read_csv(
                self.paths["siginfo_file"], sep="\t", low_memory=False
            )
            self.siginfo_df = self.siginfo.loc[
                (self.siginfo.pert_time == 24.0) & (~self.siginfo.pert_dose.isna())
            ].reset_index(drop=False)

            # Load CTRP files
            self.curves_df = pd.read_csv(
                self.paths["curves_post_qc"], sep="\t", low_memory=False
            )
            self.per_cpd_post_qc = pd.read_csv(
                self.paths["per_cpd_post_qc"], sep="\t", low_memory=False
            )
            self.per_experiment = pd.read_csv(
                self.paths["per_experiment"], sep="\t", low_memory=False
            )
            self.per_compound = pd.read_csv(
                self.paths["per_compound"], sep="\t", low_memory=False
            )
            self.per_cell_line = pd.read_csv(
                self.paths["per_cell_line"], sep="\t", low_memory=False
            )

            # Load GCTX file metadata
            with h5py.File(self.paths["gctx_file"], "r") as f:
                # Read metadata carefully
                self.gene_ids = np.array(f["0"]["META"]["ROW"]["id"]).astype(int)
                self.sig_ids = pd.Series(f["0"]["META"]["COL"]["id"]).astype(str)

                # Get matrix shape and dtype
                self.matrix_shape = f["0"]["DATA"]["0"]["matrix"].shape
                self.matrix_dtype = f["0"]["DATA"]["0"]["matrix"].dtype

            self.logger.info(f"GCTX Matrix Shape: {self.matrix_shape}")
            self.logger.info(f"GCTX Matrix Dtype: {self.matrix_dtype}")

        except Exception as e:
            self.logger.error(f"Error loading data files: {e}")
            raise

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
        Optimized matching of LINCS signatures with CTRP drug response curves.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        import numpy as np
        import pandas as pd

        self.logger.info("Starting optimized LINCS-CTRP matching process")

        # Log initial DataFrame shapes and details
        self.logger.info(f"LINCS Signatures DataFrame shape: {self.siginfo_df.shape}")
        self.logger.info(f"CTRP Curves DataFrame shape: {self.curves_df.shape}")

        # Create mappings using more robust methods
        def create_unique_mapping(df, index_col, value_col):
            """
            Create a mapping with unique values, keeping first occurrence
            """
            mapping = df.drop_duplicates(subset=[index_col])
            return pd.Series(mapping[value_col].values, index=mapping[index_col])

        # Create unique mappings
        experiment_to_ccl = create_unique_mapping(
            self.per_experiment, "experiment_id", "master_ccl_id"
        )
        ccl_to_name = create_unique_mapping(
            self.per_cell_line, "master_ccl_id", "ccl_name"
        )
        cpd_to_broad = create_unique_mapping(
            self.per_compound, "master_cpd_id", "broad_cpd_id"
        )

        # Preprocess curves DataFrame
        curves_df_processed = self.curves_df.copy()

        # Add cell line and compound names with safe mapping
        def safe_map(series, mapping):
            """
            Safely map series with a given mapping
            """
            return series.map(mapping).fillna("")

        curves_df_processed["cell_name"] = safe_map(
            safe_map(curves_df_processed["experiment_id"], experiment_to_ccl),
            ccl_to_name,
        )
        curves_df_processed["cpd_name"] = safe_map(
            curves_df_processed["master_cpd_id"], cpd_to_broad
        )

        # Drop rows with missing cell or compound names
        curves_df_processed.dropna(subset=["cell_name", "cpd_name"], inplace=True)
        curves_df_processed = curves_df_processed[
            (curves_df_processed["cell_name"] != "")
            & (curves_df_processed["cpd_name"] != "")
        ]

        # Prepare signature DataFrames
        siginfo_strategies = [
            self.siginfo_df[["cell_mfc_name", "pert_mfc_id", "sig_id", "pert_dose"]],
            self.siginfo_df[["cell_mfc_name", "cmap_name", "sig_id", "pert_dose"]],
        ]

        # Storage for matched data
        all_matches = {}
        all_smiles = []
        all_viabilities = []
        all_sig_ids = []

        # Matching function
        def process_strategy(strategy_df):
            local_matches = {}
            local_smiles = []
            local_viabilities = []
            local_sig_ids = []

            # Vectorized matching
            for _, row in strategy_df.iterrows():
                cell_name, compound_name = (
                    row["cell_mfc_name"],
                    row[strategy_df.columns[1]],
                )

                # Find matching rows in curves DataFrame
                matching_curves = curves_df_processed[
                    (curves_df_processed["cell_name"] == cell_name)
                    & (curves_df_processed["cpd_name"] == compound_name)
                ]

                if not matching_curves.empty:
                    for _, curve in matching_curves.iterrows():
                        try:
                            # Compute viability
                            dose = row["pert_dose"]
                            curve_params = [
                                curve["p1_center"],
                                curve["p2_slope"],
                                curve["p4_baseline"],
                            ]

                            viability = self._sig3upper(curve_params, np.log2(dose))

                            # Get SMILES
                            smile = self.per_compound.loc[
                                self.per_compound.master_cpd_id
                                == curve["master_cpd_id"],
                                "cpd_smiles",
                            ].values[0]

                            local_matches[row["sig_id"]] = curve.name
                            local_smiles.append(smile)
                            local_viabilities.append(viability)
                            local_sig_ids.append(row["sig_id"])

                        except Exception as e:
                            self.logger.warning(f"Match processing error: {e}")
                            continue

            return local_matches, local_smiles, local_viabilities, local_sig_ids

        # Process strategies sequentially to ensure unique sig_ids
        for strategy_df in siginfo_strategies:
            matches, smiles, viabilities, sig_ids = process_strategy(strategy_df)

            # Update only non-duplicate matches
            for sig_id, match in matches.items():
                if sig_id not in all_matches:
                    all_matches[sig_id] = match
                    idx = sig_ids.index(sig_id)
                    all_smiles.append(smiles[idx])
                    all_viabilities.append(viabilities[idx])
                    all_sig_ids.append(sig_id)

        # Log results
        self.logger.info(f"Total matches found: {len(all_matches)}")
        self.logger.info(f"Total unique compounds: {len(set(all_smiles))}")

        return all_matches, all_smiles, all_viabilities, all_sig_ids

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

        with h5py.File(self.paths["gctx_file"], "r") as f:
            matrix_dataset = f["0"]["DATA"]["0"]["matrix"]

            for i in range(0, len(row_indices), chunk_size):
                chunk_indices = row_indices[i : i + chunk_size]
                chunk = matrix_dataset[chunk_indices, :]
                retrieved_rows.append(chunk)

        return np.concatenate(retrieved_rows, axis=0)

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
            expression_file=self.paths["gctx_file"],
            matrix_shape=matched_matrix.shape,
            row_metadata_file=self.paths["siginfo_file"],
            compound_info_file=self.paths["per_compound"],
            gene_info_file=self.paths["geneinfo_file"],
            metadata=metadata,
        )

        # Override internal data
        adapter.expression_data = matched_matrix
        adapter.row_metadata = matched_siginfo
        adapter.row_ids = list(matched_siginfo.index)
        adapter.col_metadata = pd.read_csv(self.paths["geneinfo_file"], sep="\t")
        adapter.col_ids = self.gene_ids.tolist()
        adapter.col_symbols = self.geneinfo_df.loc[self.gene_ids].gene_symbol.tolist()

        # Write to file
        adapter.write_data(self.output_path)

        self.logger.info(f"Matched dataset written to {self.output_path}")
        self.logger.info(f"Total matched signatures: {len(sig_ids)}")


def main():
    """
    Main execution for LINCS-CTRP dataset matching.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("lincs_ctrp_matcher.log"),
        ],
    )

    logger = logging.getLogger(__name__)

    try:
        # Determine config path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_path = os.path.join(project_root, "config.yaml")

        logger.info(f"Loading configuration from {config_path}")

        # Load configuration
        config = load_config(config_path)

        # Optional: Override specific config parameters via CLI
        cli_overrides = parse_cli_overrides()
        if cli_overrides:
            config = merge_configs(config, cli_overrides)

        # Create matcher
        matcher = LINCSCTRPMatcher(config)

        # Time the matching process
        import time

        start_time = time.time()

        # Perform matching and write dataset
        matcher.match_and_write()

        end_time = time.time()
        logger.info(
            f"Matching process completed in {end_time - start_time:.2f} seconds"
        )

    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in LINCS-CTRP matching process: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
