# src/data/matcher.py
"""
Module for matching LINCS and CTRP datasets and writing to .gctx format.
"""

import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from ..config.config_utils import load_config, merge_configs, parse_cli_overrides
from ..config.constants import CURRENT_SCHEMA_VERSION, REQUIRED_CONFIG_SECTIONS
from ..data.adapters import DatasetMetadata, LINCSAdapter


class LINCSCTRPMatcher:
    """
    Advanced matcher for LINCS and CTRP datasets with optimized performance and progress tracking.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the matcher with validated configuration.

        Args:
            config: Validated configuration dictionary
            logger: Optional custom logger
        """
        self._validate_config_sections(config)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Setup data paths from configuration
        self.paths = self._setup_paths()
        self.output_path = self.config["data"].get(
            "output_path",
            os.path.join(
                os.path.dirname(self.paths["gctx_file"]),
                "..",
                "processed",
                "LINCS_CTRP_matched.gctx",
            ),
        )

        # Data containers
        self._initialize_data_containers()

        # Checkpoint file
        self.checkpoint_file = "matches_checkpoint.pkl"

    def _validate_config_sections(self, config: Dict[str, Any]):
        """Validate that all required configuration sections are present."""
        missing_sections = REQUIRED_CONFIG_SECTIONS - set(config.keys())
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")

    def _setup_paths(self) -> Dict[str, str]:
        """Extract and validate file paths from configuration."""
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
        paths = {file_key: data_config.get(file_key) for file_key in required_files}
        for key, path in paths.items():
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"File not found for {key}: {path}")
        return paths

    def _initialize_data_containers(self):
        """Initialize data containers with None."""
        containers = [
            "geneinfo",
            "siginfo",
            "curves_df",
            "per_cpd_post_qc",
            "per_experiment",
            "per_compound",
            "per_cell_line",
            "gene_ids",
            "sig_ids",
            "matrix_shape",
            "matrix_dtype",
        ]
        for container in containers:
            setattr(self, container, None)

    def load_data(self):
        """Load all necessary data files with robust error handling."""
        self.logger.info("Loading data files...")
        try:
            pd.options.mode.chained_assignment = None  # Suppress warnings
            # LINCS data
            self.geneinfo = pd.read_csv(
                self.paths["geneinfo_file"], sep="\t", low_memory=False
            )
            self.geneinfo_df = self.geneinfo.set_index("gene_id")
            self.siginfo = pd.read_csv(
                self.paths["siginfo_file"], sep="\t", low_memory=False
            )
            self.siginfo_df = self.siginfo.loc[
                (self.siginfo.pert_time == 24.0) & (~self.siginfo.pert_dose.isna())
            ].reset_index(drop=False)
            # CTRP data
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
            # GCTX metadata
            with h5py.File(self.paths["gctx_file"], "r") as f:
                self.gene_ids = np.array(f["0"]["META"]["ROW"]["id"]).astype(int)
                self.sig_ids = pd.Series(f["0"]["META"]["COL"]["id"]).astype(str)
                self.matrix_shape = f["0"]["DATA"]["0"]["matrix"].shape
                self.matrix_dtype = f["0"]["DATA"]["0"]["matrix"].dtype
            self.logger.info(
                f"GCTX Matrix Shape: {self.matrix_shape}, Dtype: {self.matrix_dtype}"
            )
        except Exception as e:
            self.logger.error(f"Error loading data files: {e}")
            raise

    def _sig3upper(self, p: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Dose-response curve implementation from CTRP."""
        alpha, beta, b = p.T  # Transpose for vectorized operation
        exponent = np.clip(-(X - alpha) / beta, -100, 100)
        return b + (1 - b) / (1 + np.exp(exponent))

    def _process_strategy_chunk(
        self, chunk: pd.DataFrame, curves_df_processed: pd.DataFrame
    ) -> pd.DataFrame:
        """Process a chunk of signatures for matching."""
        # Vectorized merge
        merged = chunk.merge(
            curves_df_processed,
            left_on=["cell_mfc_name", chunk.columns[1]],
            right_on=["cell_name", "cpd_name"],
            how="inner",
        )
        if merged.empty:
            return pd.DataFrame(columns=["sig_id", "viability", "smiles"])

        # Vectorized viability calculation
        curve_params = merged[["p1_center", "p2_slope", "p4_baseline"]].values
        merged["viability"] = self._sig3upper(
            curve_params, np.log2(merged["pert_dose"].values)
        )

        # Map SMILES
        smiles_map = self.per_compound.set_index("master_cpd_id")["cpd_smiles"]
        merged["smiles"] = merged["master_cpd_id"].map(smiles_map)

        # Validate SMILES
        valid_mask = merged["smiles"].apply(lambda x: Chem.MolFromSmiles(x) is not None)
        invalid_count = (~valid_mask).sum()
        if invalid_count:
            self.logger.warning(f"Found {invalid_count} invalid SMILES in chunk")

        return merged[valid_mask][["sig_id", "viability", "smiles"]]

    def _match_siginfo_and_ctrp(self) -> Tuple[Dict, List[str], List[float], List[str]]:
        """Optimized matching of LINCS signatures with CTRP drug response curves."""
        self.logger.info("Starting optimized LINCS-CTRP matching process")
        self.logger.info(
            f"LINCS Signatures: {self.siginfo_df.shape}, CTRP Curves: {self.curves_df.shape}"
        )

        # Preprocess mappings
        exp_to_ccl = self.per_experiment.drop_duplicates("experiment_id").set_index(
            "experiment_id"
        )["master_ccl_id"]
        ccl_to_name = self.per_cell_line.drop_duplicates("master_ccl_id").set_index(
            "master_ccl_id"
        )["ccl_name"]
        cpd_to_broad = self.per_compound.drop_duplicates("master_cpd_id").set_index(
            "master_cpd_id"
        )["broad_cpd_id"]

        curves_df_processed = self.curves_df.copy()
        curves_df_processed["cell_name"] = (
            curves_df_processed["experiment_id"].map(exp_to_ccl).map(ccl_to_name)
        )
        curves_df_processed["cpd_name"] = curves_df_processed["master_cpd_id"].map(
            cpd_to_broad
        )
        curves_df_processed.dropna(subset=["cell_name", "cpd_name"], inplace=True)

        # Matching strategies
        siginfo_strategies = [
            self.siginfo_df[["cell_mfc_name", "pert_mfc_id", "sig_id", "pert_dose"]],
            self.siginfo_df[
                ["cell_mfc_name", "cmap_name", "sig_id", "pert_dose"]
            ].rename(columns={"cmap_name": "pert_mfc_id"}),
        ]

        all_matches, all_smiles, all_viabilities, all_sig_ids = {}, [], [], []
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "rb") as f:
                all_matches, all_smiles, all_viabilities, all_sig_ids = pickle.load(f)
            self.logger.info(f"Loaded checkpoint with {len(all_matches)} matches")

        chunk_size = 1000
        with ProcessPoolExecutor() as executor:
            for strategy_idx, sig_df in enumerate(siginfo_strategies):
                chunks = [
                    sig_df[i : i + chunk_size]
                    for i in range(0, len(sig_df), chunk_size)
                ]
                futures = {
                    executor.submit(
                        self._process_strategy_chunk, chunk, curves_df_processed
                    ): i
                    for i, chunk in enumerate(chunks)
                    if not all(chunk["sig_id"].isin(all_sig_ids))
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Strategy {strategy_idx+1}",
                ):
                    try:
                        merged = future.result()
                        for _, row in merged.iterrows():
                            sig_id = row["sig_id"]
                            if sig_id not in all_matches:
                                all_matches[sig_id] = (
                                    None  # Could store curve index if needed
                                )
                                all_smiles.append(row["smiles"])
                                all_viabilities.append(row["viability"])
                                all_sig_ids.append(sig_id)
                        # Checkpoint every 10 chunks
                        if futures[future] % 10 == 0:
                            with open(self.checkpoint_file, "wb") as f:
                                pickle.dump(
                                    (
                                        all_matches,
                                        all_smiles,
                                        all_viabilities,
                                        all_sig_ids,
                                    ),
                                    f,
                                )
                            self.logger.info(
                                f"Checkpoint saved: {len(all_matches)} matches"
                            )
                    except Exception as e:
                        self.logger.error(f"Chunk processing error: {e}")

        (
            os.remove(self.checkpoint_file)
            if os.path.exists(self.checkpoint_file)
            else None
        )
        self.logger.info(
            f"Total matches: {len(all_matches)}, Unique compounds: {len(set(all_smiles))}"
        )
        return all_matches, all_smiles, all_viabilities, all_sig_ids

    def _retrieve_matrix_rows(self, row_indices: List[int]) -> np.ndarray:
        """Retrieve specific rows from the GCTX matrix in chunks."""
        max_rows = self.matrix_shape[0]
        if any(idx >= max_rows for idx in row_indices):
            raise ValueError(f"Row indices exceed matrix dimensions: {max_rows}")

        chunk_size = 10000
        retrieved_rows = []
        with h5py.File(self.paths["gctx_file"], "r") as f:
            matrix_dataset = f["0"]["DATA"]["0"]["matrix"]
            for i in tqdm(
                range(0, len(row_indices), chunk_size), desc="Retrieving matrix rows"
            ):
                chunk_indices = row_indices[i : i + chunk_size]
                chunk = matrix_dataset[chunk_indices, :]
                retrieved_rows.append(chunk)
        return np.concatenate(retrieved_rows, axis=0)

    def match_and_write(self):
        """Perform matching and write to .gctx file."""
        start_time = time.time()
        self.load_data()
        matches, smiles, viabilities, sig_ids = self._match_siginfo_and_ctrp()

        # Metadata
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

        # Matrix rows
        matrix_indices = [
            np.where(self.sig_ids == sig_id)[0][0]
            for sig_id in tqdm(sig_ids, desc="Indexing signatures")
        ]
        matched_matrix = self._retrieve_matrix_rows(matrix_indices)

        # Signature metadata
        matched_siginfo = self.siginfo_df.set_index("sig_id").loc[sig_ids].reset_index()
        matched_siginfo["smiles"] = smiles
        matched_siginfo["viability"] = viabilities
        matched_siginfo.set_index("sig_id", inplace=True)

        # Use LINCSAdapter for writing
        adapter = LINCSAdapter(
            expression_file=self.paths["gctx_file"],
            matrix_shape=matched_matrix.shape,
            row_metadata_file=self.paths["siginfo_file"],
            compound_info_file=self.paths["per_compound"],
            gene_info_file=self.paths["geneinfo_file"],
            metadata=metadata,
        )
        adapter.expression_data = matched_matrix
        adapter.row_metadata = matched_siginfo
        adapter.row_ids = list(matched_siginfo.index)
        adapter.col_metadata = self.geneinfo
        adapter.col_ids = self.gene_ids.tolist()
        adapter.col_symbols = self.geneinfo_df.loc[self.gene_ids].gene_symbol.tolist()

        adapter.write_data(self.output_path)
        self.logger.info(f"Matched dataset written to {self.output_path}")
        self.logger.info(f"Total matched signatures: {len(sig_ids)}")
        self.logger.info(f"Process completed in {time.time() - start_time:.2f} seconds")


def main():
    """Main execution for LINCS-CTRP dataset matching."""
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_path = os.path.join(project_root, "config.yaml")
        logger.info(f"Loading configuration from {config_path}")

        config = load_config(config_path)
        cli_overrides = parse_cli_overrides()
        if cli_overrides:
            config = merge_configs(config, cli_overrides)

        matcher = LINCSCTRPMatcher(config)
        matcher.match_and_write()

    except Exception as e:
        logger.error(f"Error in matcher: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
