# src/data/adapters.py
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""

    version: str
    description: str
    data_source: str
    creation_date: str
    platform: str
    additional_info: Optional[Dict] = None


class DataAdapter(ABC):
    """Abstract base class for data adapters."""

    @abstractmethod
    def read_data(self) -> Tuple:
        """Read data from source files."""
        pass

    @abstractmethod
    def process_data(self) -> Tuple:
        """Process and clean the data."""
        pass

    @abstractmethod
    def write_data(self, output_path: str) -> None:
        """Write processed data to output file."""
        pass

    def convert(self, output_path: str) -> None:
        """Convert data from source to destination format."""
        try:
            logger.info("Starting data conversion process")
            data = self.read_data()
            processed_data = self.process_data()
            self.write_data(output_path)
            logger.info(f"Data conversion completed: {output_path}")
        except Exception as e:
            logger.error(f"Error during data conversion: {str(e)}")
            raise


class LINCSAdapter(DataAdapter):
    """Adapter for LINCS data format conversion."""

    def __init__(
        self,
        expression_file: str,
        matrix_shape: Tuple[int, int],
        row_metadata_file: str,
        compound_info_file: str,
        gene_info_file: str,
        metadata: DatasetMetadata,
    ):
        """
        Initialize the LINCS adapter.

        Args:
            expression_file: Path to gene expression data file
            row_metadata_file: Path to row metadata file
            compound_info_file: Path to compound information file
            gene_info_file: Path to gene information file
            metadata: Dataset metadata information
        """
        self.expression_file = expression_file
        self.matrix_shape = matrix_shape
        self.row_metadata_file = row_metadata_file
        self.compound_info_file = compound_info_file
        self.gene_info_file = gene_info_file
        self.metadata = metadata

        # Initialize data containers
        self.expression_data = None
        self.row_metadata = None
        self.compound_info = None
        self.col_metadata = None
        self.row_ids = None
        self.col_ids = None

    def read_data(self) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Read data from source files.

        Returns:
            Tuple containing expression data, row metadata, compound info, and column metadata
        """
        logger.info("Loading gene expression data from binary file...")
        try:
            # Load gene expression data
            self.expression_data = np.memmap(
                self.expression_file, dtype=np.float64, mode="r"
            ).reshape(self.matrix_shape)

            logger.info(f"Gene expression data shape: {self.expression_data.shape}")

            # Load row metadata
            logger.info("Loading row metadata...")
            self.row_metadata = pd.read_csv(self.row_metadata_file, sep="\t")

            # Load compound info
            logger.info("Loading compound info...")
            self.compound_info = pd.read_csv(self.compound_info_file)
            self.compound_info = self.compound_info[
                ["pert_id", "cmap_name", "canonical_smiles"]
            ]
            self.compound_info.set_index("pert_id", inplace=True)
            self.compound_info = self.compound_info[
                ~self.compound_info.index.duplicated(keep="first")
            ]
            logger.info(
                f"Compound info shape after dropping duplicates: {self.compound_info.shape}"
            )

            # Load column metadata
            logger.info("Loading column metadata...")
            self.col_metadata = pd.read_csv(self.gene_info_file, sep="\t")

            return (
                self.expression_data,
                self.row_metadata,
                self.compound_info,
                self.col_metadata,
            )

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise

    def process_data(self) -> Tuple[np.ndarray, pd.DataFrame, List[str], List[str]]:
        """
        Process and clean the data.

        Returns:
            Tuple containing processed expression data, row metadata, row IDs, and column IDs
        """
        logger.info("Processing data...")

        # Validate shapes
        assert (
            self.expression_data.shape == self.matrix_shape
        ), f"Expected expression data shape {self.matrix_shape}, got {self.expression_data.shape}"
        assert (
            self.row_metadata.shape[0] == self.matrix_shape[0]
        ), f"Expected 31567 rows in row metadata, got {self.row_metadata.shape[0]}"
        assert self.expression_data.shape[1] == self.col_metadata.shape[0], (
            f"Mismatch in columns: {self.expression_data.shape[1]} (expression) vs. "
            f"{self.col_metadata.shape[0]} (column metadata)"
        )

        # Merge row metadata with compound info
        self.row_metadata = pd.merge(
            self.row_metadata,
            self.compound_info,
            left_on="pert_mfc_id",
            right_index=True,
            how="left",
        )

        # Drop rows with NaN values
        mask_metadata = self.row_metadata.notnull().all(axis=1).values
        mask_expr = ~np.any(np.isnan(self.expression_data), axis=1)
        final_mask = mask_metadata & mask_expr
        num_dropped = np.sum(~final_mask)
        logger.info(f"Dropping {num_dropped} rows due to NaN values.")

        self.row_metadata = self.row_metadata.loc[final_mask].copy()
        self.expression_data = self.expression_data[final_mask, :]

        # Prepare row and column IDs
        self.row_metadata.set_index("sig_id", inplace=True)
        self.row_ids = self.row_metadata.index.astype(str).tolist()
        self.col_ids = self.col_metadata["gene_id"].astype(str).tolist()
        self.col_symbols = self.col_metadata["gene_symbol"].tolist()

        logger.info(f"Final processed data shape: {self.expression_data.shape}")
        return self.expression_data, self.row_metadata, self.row_ids, self.col_ids

    def write_data(self, output_path: str) -> None:
        """
        Write processed data to .gctx file.

        Args:
            output_path: Path to output file
        """
        logger.info(f"Writing data to {output_path}")

        # Make sure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        n_rows_out = len(self.row_ids)
        n_cols_out = self.expression_data.shape[1]
        chunk_shape_out = (256, n_cols_out)

        with h5py.File(output_path, "w") as f_out:
            # Write expression data
            grp_data = f_out.create_group("0/DATA/0")
            dset = grp_data.create_dataset(
                "matrix",
                shape=(n_rows_out, n_cols_out),
                dtype="float32",
                compression="gzip",
                chunks=chunk_shape_out,
            )

            # Write in chunks to handle large datasets
            chunk_rows_read = 16384
            out_row = 0
            for start in range(0, n_rows_out, chunk_rows_read):
                end = min(start + chunk_rows_read, n_rows_out)
                chunk_data = self.expression_data[start:end, :].astype("float32")
                dset[out_row : out_row + (end - start), :] = chunk_data
                out_row += end - start

            logger.info("Gene expression data written")

            # Write row metadata
            meta_row_grp = f_out.create_group("0/META/ROW")
            for col in self.row_metadata.columns:
                data = self.row_metadata[col].values
                if pd.api.types.is_numeric_dtype(self.row_metadata[col]):
                    meta_row_grp.create_dataset(col, data=data)
                else:
                    meta_row_grp.create_dataset(col, data=data.astype(str).astype("S"))
            meta_row_grp.create_dataset("id", data=np.array(self.row_ids, dtype="S"))

            # Write column metadata
            meta_col_grp = f_out.create_group("0/META/COL")
            for col in self.col_metadata.columns:
                data = self.col_metadata[col].values
                if pd.api.types.is_numeric_dtype(self.col_metadata[col]):
                    meta_col_grp.create_dataset(col, data=data)
                else:
                    meta_col_grp.create_dataset(col, data=data.astype(str).astype("S"))
            meta_col_grp.create_dataset(
                "id", data=np.array(self.col_symbols, dtype="S")
            )

            # Add metadata attributes
            f_out.attrs["version"] = self.metadata.version
            f_out.attrs["description"] = self.metadata.description
            f_out.attrs["data_source"] = self.metadata.data_source
            f_out.attrs["creation_date"] = self.metadata.creation_date
            f_out.attrs["platform"] = self.metadata.platform

            if self.metadata.additional_info:
                for key, value in self.metadata.additional_info.items():
                    f_out.attrs[key] = value

        logger.info(f"Data successfully written to {output_path}")


class LINCSDatasetFactory:
    """Factory for creating and converting LINCS datasets."""

    @staticmethod
    def create_dataset(config: Dict[str, str]) -> LINCSAdapter:
        """
        Create a LINCS dataset adapter from configuration.

        Args:
            config: Configuration dictionary containing file paths and metadata

        Returns:
            LINCSAdapter instance
        """
        required_keys = [
            "expression_file",
            "row_metadata_file",
            "compound_info_file",
            "gene_info_file",
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        metadata = DatasetMetadata(
            version=config.get("version", "1.0"),
            description=config.get(
                "description", "LINCS gene expression data in .gctx format"
            ),
            data_source=config.get("data_source", "CLUE.io, CTRPv2"),
            creation_date=pd.Timestamp.now().isoformat(),
            platform=config.get("platform", "L1000"),
            additional_info=config.get("additional_info"),
        )

        return LINCSAdapter(
            expression_file=config["expression_file"],
            matrix_shape=config["matrix_shape"],
            row_metadata_file=config["row_metadata_file"],
            compound_info_file=config["compound_info_file"],
            gene_info_file=config["gene_info_file"],
            metadata=metadata,
        )

    @staticmethod
    def convert_dataset(config: Dict[str, str], output_path: str) -> None:
        """
        Convert a LINCS dataset using the configuration provided.

        Args:
            config: Configuration dictionary
            output_path: Path to output file
        """
        adapter = LINCSDatasetFactory.create_dataset(config)
        adapter.convert(output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Configuration
    config = {
        "expression_file": "../data/raw/X_RNA.bin",
        "matrix_shape": (31567, 12328),
        "row_metadata_file": "../data/processed/Y.tsv",
        "compound_info_file": "../data/raw/compoundinfo.csv",
        "gene_info_file": "../data/raw/LINCS/geneinfo_beta.txt",
        "description": "LINCS gene expression data in .gctx format combined with CTRPv2 cell viability data.",
        "data_source": "CLUE.io, CTRPv2",
        "platform": "L1000",
        "additional_info": {
            "num_cell_lines": 30,
            "num_compounds": 500,
        },
    }

    # Convert dataset
    output_path = "../data/processed/LINCS.gctx"
    LINCSDatasetFactory.convert_dataset(config, output_path)
