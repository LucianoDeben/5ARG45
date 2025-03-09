# src/data/datasets.py
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, TypeVar

import numpy as np
import pandas as pd
from src.data.feature_transforms import MorganFingerprintTransform
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)

# Type variable for dataset classes
T = TypeVar('T', bound=Dataset)


class MultimodalDrugDataset(Dataset):
    """Dataset for multimodal drug response prediction combining transcriptomics and molecular data."""
    
    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        metadata: pd.DataFrame,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
    ):
        """
        Initialize the multimodal dataset.
        
        Args:
            transcriptomics_data: Array of gene expression values
            metadata: DataFrame with drug and experiment metadata
            transform_transcriptomics: Optional transform for transcriptomics data
            transform_molecular: Optional transform for molecular data
        """
        self.transcriptomics_data = transcriptomics_data
        self.metadata = metadata
        self.transform_transcriptomics = transform_transcriptomics
        self.transform_molecular = transform_molecular

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get transcriptomics data
        transcriptomics = torch.tensor(
            self.transcriptomics_data[idx], dtype=torch.float32
        )
        if self.transform_transcriptomics:
            transcriptomics = self.transform_transcriptomics(transcriptomics)

        # Get molecular data
        row = self.metadata.iloc[idx]
        smiles = row["canonical_smiles"]
        dosage = torch.tensor(row["pert_dose_log2"], dtype=torch.float32)

        molecular = None
        if self.transform_molecular:
            mol_input = {"smiles": [smiles], "dosage": [dosage]}
            molecular = self.transform_molecular(mol_input)
            # Ensure we get the first element if transform returns a batch
            if isinstance(molecular, list) and len(molecular) > 0:
                molecular = molecular[0]
            molecular = torch.from_numpy(molecular) if not isinstance(molecular, torch.Tensor) else molecular

        viability = torch.tensor(row["viability_clipped"], dtype=torch.float32)

        return {
            "transcriptomics": transcriptomics,
            "molecular": molecular,
            "dosage": dosage,
            "viability": viability,
        }


class TranscriptomicsDataset(Dataset):
    """Dataset for transcriptomics-based drug response prediction."""
    
    def __init__(
        self,
        transcriptomics_data: np.ndarray,
        viability: np.ndarray,
        transform_transcriptomics: Optional[Callable] = None,
    ):
        """
        Initialize the transcriptomics dataset.
        
        Args:
            transcriptomics_data: Array of gene expression values
            viability: Array of cell viability values
            transform_transcriptomics: Optional transform for transcriptomics data
        """
        self.transcriptomics_data = transcriptomics_data
        self.viability = viability
        self.transform_transcriptomics = transform_transcriptomics

    def __len__(self) -> int:
        return len(self.transcriptomics_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.transcriptomics_data[idx], dtype=torch.float32)
        y = torch.tensor(self.viability[idx], dtype=torch.float32)
        if self.transform_transcriptomics:
            x = self.transform_transcriptomics(x)
        return x, y


class ChemicalDataset(Dataset):
    """Dataset for chemical-based drug response prediction."""
    
    def __init__(
        self,
        smiles: np.ndarray,
        dosage: np.ndarray,
        viability: np.ndarray,
        transform_molecular: Optional[Callable] = None,
    ):
        """
        Initialize the chemical dataset.
        
        Args:
            smiles: Array of SMILES strings for drugs
            dosage: Array of drug dosages
            viability: Array of cell viability values
            transform_molecular: Optional transform for chemical data
        """
        self.smiles = smiles
        self.dosage = dosage
        self.viability = viability
        self.transform_molecular = transform_molecular

    def __len__(self) -> int:
        return len(self.smiles)

    #TODO: Optimal to never have SMILES strings in the dataset, but rather the processed molecular data, check if this can be enforced
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Convert SMILES to a tensor or representation immediately
        smiles = self.smiles[idx]
        molecular = None
        if self.transform_molecular:
            molecular = self.transform_molecular(smiles)
        
        return {
            "molecular": molecular,
            "dosage": torch.tensor(self.dosage[idx], dtype=torch.float32),
            "viability": torch.tensor(self.viability[idx], dtype=torch.float32),
        }

#TODO: Confirm that the factory actually selectes the correct metadata based on the data matrix used for both efficiency and correctness
class DatasetFactory:
    """Factory for creating and splitting datasets from GCTX data with chunking support."""

    @staticmethod
    def _validate_split_sizes(test_size: float, val_size: float) -> None:
        """Validate that test and validation sizes are within acceptable ranges."""
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"Combined test ({test_size}) and validation ({val_size}) sizes must be less than 1.0"
            )

    @staticmethod
    def _validate_required_columns(metadata: pd.DataFrame, required_cols: List[str]) -> None:
        """Validate that required columns are present in metadata."""
        missing = [col for col in required_cols if col not in metadata.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _load_metadata(
        gctx_loader: GCTXDataLoader,
        nrows: Optional[int] = None,
        chunk_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Load metadata from GCTX loader with chunking support.
        
        Args:
            gctx_loader: GCTX data loader
            nrows: Number of rows to load
            chunk_size: Size of chunks for processing

        Returns:
            Metadata DataFrame
        """
        logger.info("Loading metadata...")
        
        with gctx_loader:
            total_rows = (
                gctx_loader._n_rows
                if nrows is None
                else min(nrows, gctx_loader._n_rows)
            )

            # Process in chunks if dataset is large
            if total_rows > chunk_size:
                metadata_chunks = []
                for start in tqdm(
                    range(0, total_rows, chunk_size), desc="Loading metadata"
                ):
                    end = min(start + chunk_size, total_rows)
                    chunk = gctx_loader.get_row_metadata(row_slice=slice(start, end))
                    metadata_chunks.append(chunk)

                metadata = pd.concat(metadata_chunks, ignore_index=True)
            else:
                # Load entire dataset if small enough
                metadata = gctx_loader.get_row_metadata(row_slice=slice(0, total_rows))

        # Ensure metadata is a DataFrame
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)
            
        logger.info(f"Loaded metadata with {len(metadata)} rows")
        return metadata

    @staticmethod
    def _split_data(
        metadata: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split indices based on metadata with support for large datasets.
        Supports both group-based splitting and stratification.

        Args:
            metadata: DataFrame containing metadata
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            group_by: Column to use for group-based splitting
            stratify_by: Column to use for stratified splitting

        Returns:
            Tuple of train, validation, and test indices
        """
        logger.info("Performing data splitting...")
        
        # Validate split sizes
        DatasetFactory._validate_split_sizes(test_size, val_size)
        
        # Calculate theoretical split sizes for reference
        n_samples = len(metadata)
        expected_test_size = int(n_samples * test_size)
        expected_val_size = int(n_samples * val_size)
        expected_train_size = n_samples - expected_test_size - expected_val_size
        
        logger.info(f"Expected split: Train={expected_train_size}, Val={expected_val_size}, Test={expected_test_size}")

        # Generate array of indices
        indices = np.arange(n_samples)
        
        # Modify the splitting order to ensure test > val in size
        # Now we'll split: train vs (val+test), then val vs test
        combined_val_test_size = test_size + val_size
        
        # Prepare stratification if needed
        stratify = None
        if stratify_by and stratify_by in metadata.columns:
            try:
                if metadata[stratify_by].dtype in [float, int]:
                    # For continuous variables, create bins
                    stratify = pd.qcut(
                        metadata[stratify_by], q=10, labels=False, duplicates="drop"
                    )
                else:
                    stratify = metadata[stratify_by]
                logger.info(f"Using stratified splitting based on '{stratify_by}'")
            except Exception as e:
                logger.warning(f"Stratification failed: {e}. Using random splitting.")
                stratify = None

        # Handle group-based splitting
        if group_by and group_by in metadata.columns and len(metadata[group_by].unique()) >= 3:
            logger.info(f"Using group-based splitting based on '{group_by}'")
            
            # If also using stratification, we'll try an approach that balances both constraints
            if stratify is not None:
                try:
                    # Create a summary of each group's distribution of the stratification variable
                    group_summaries = []
                    for group_name, group_data in metadata.groupby(group_by):
                        # For each group, get its size and average stratification value
                        if isinstance(stratify, pd.Series):
                            group_strat_value = stratify.loc[group_data.index].mean()
                        else:
                            group_strat_value = stratify[group_data.index].mean()
                        
                        group_summaries.append({
                            'group': group_name,
                            'size': len(group_data),
                            'strat_value': group_strat_value,
                            'indices': group_data.index.tolist()
                        })
                    
                    # Sort groups by stratification value to help with balanced assignment
                    group_summaries.sort(key=lambda x: x['strat_value'])
                    
                    # Assign groups to ensure size constraints while balancing stratification
                    train_indices = []
                    val_test_indices = []
                    
                    # Use a greedy approach to assign groups
                    current_train_size = 0
                    target_train_size = n_samples * (1 - combined_val_test_size)
                    
                    # Alternate taking from beginning and end of sorted list to maintain distribution
                    take_from_start = True
                    
                    while group_summaries:
                        if take_from_start:
                            group = group_summaries.pop(0)
                        else:
                            group = group_summaries.pop(-1)
                            
                        take_from_start = not take_from_start
                        
                        if current_train_size < target_train_size:
                            train_indices.extend(group['indices'])
                            current_train_size += group['size']
                        else:
                            val_test_indices.extend(group['indices'])
                    
                    # Now split val_test_indices into val and test
                    # Calculate adjusted sizes based on what we have
                    val_test_size = len(val_test_indices)
                    adjusted_test_size = test_size / combined_val_test_size
                    
                    # Stratify again for the val vs test split
                    val_test_strat = None
                    if stratify is not None:
                        if isinstance(stratify, pd.Series):
                            val_test_strat = stratify.loc[val_test_indices]
                        else:
                            val_test_strat = stratify[val_test_indices]
                    
                    val_indices, test_indices = train_test_split(
                        val_test_indices,
                        test_size=adjusted_test_size,
                        random_state=random_state,
                        stratify=val_test_strat
                    )
                    
                    # Convert to numpy arrays
                    train_idx = np.array(train_indices)
                    val_idx = np.array(val_indices)
                    test_idx = np.array(test_indices)
                    
                    logger.info("Used group-based splitting with stratification balancing")
                    
                except Exception as e:
                    logger.warning(f"Group-based stratified splitting failed: {e}. Falling back to standard group-based splitting.")
                    stratify = None
                    # Continue to regular group-based splitting
            
            # If we haven't returned from the stratified approach, use regular group-based splitting
            if stratify is None:
                try:
                    # First split: train vs (val+test)
                    gss = GroupShuffleSplit(
                        n_splits=1, test_size=combined_val_test_size, random_state=random_state
                    )
                    train_idx, val_test_idx = next(
                        gss.split(indices, groups=metadata[group_by])
                    )
                    
                    # Second split: val vs test within (val+test)
                    # Adjust sizes for the second split
                    adjusted_test_size = test_size / combined_val_test_size
                    
                    # Get groups for the val+test set
                    val_test_groups = metadata.iloc[val_test_idx][group_by].values
                    
                    # Split val+test into val and test
                    gss_test = GroupShuffleSplit(
                        n_splits=1, test_size=adjusted_test_size, random_state=random_state
                    )
                    
                    # Create indices for the val+test subset
                    val_test_indices = np.arange(len(val_test_idx))
                    
                    # Split these indices
                    val_sub_idx, test_sub_idx = next(
                        gss_test.split(val_test_indices, groups=val_test_groups)
                    )
                    
                    # Map back to original indices
                    val_idx = val_test_idx[val_sub_idx]
                    test_idx = val_test_idx[test_sub_idx]
                    
                except Exception as e:
                    logger.warning(f"Group-based splitting failed: {e}. Falling back to standard splitting.")
                    group_by = None
        
        # If not using group-based splitting or if it failed, use standard splitting
        if group_by is None or group_by not in metadata.columns:
            # First split: train vs (val+test)
            train_idx, val_test_idx = train_test_split(
                indices, 
                test_size=combined_val_test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # Second split: val vs test
            stratify_val_test = None
            if stratify is not None:
                # Use the stratification values corresponding to val_test indices
                if isinstance(stratify, pd.Series):
                    stratify_val_test = stratify.iloc[val_test_idx]
                else:
                    stratify_val_test = stratify[val_test_idx]
            
            # Get the proper test size relative to val+test
            adjusted_test_size = test_size / combined_val_test_size
            
            # Split val+test into val and test
            val_idx, test_idx = train_test_split(
                val_test_idx,
                test_size=adjusted_test_size,
                random_state=random_state,
                stratify=stratify_val_test
            )
        
        # Ensure indices are numpy arrays
        train_idx = np.array(train_idx, dtype=int)
        val_idx = np.array(val_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)
        
        # Verify the split sizes
        logger.info(f"Splitting complete. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Verify that no indices are duplicated between sets
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        
        if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
            logger.error("Split sets contain overlapping indices!")
        
        # Verify that all indices are used
        all_indices_set = train_set.union(val_set).union(test_set)
        if len(all_indices_set) != n_samples:
            logger.warning(f"Not all indices are used in the split. Expected {n_samples}, got {len(all_indices_set)}")
        
        return train_idx, val_idx, test_idx

    @staticmethod
    def _process_large_dataset(
        gctx_loader: GCTXDataLoader,
        nrows: Optional[int] = None,
        feature_space: Union[str, List[str]] = "landmark",
        chunk_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Process large datasets in chunks to manage memory.

        Args:
            gctx_loader: GCTX data loader
            nrows: Number of rows to load
            feature_space: Gene feature space
            chunk_size: Size of chunks to process

        Returns:
            Processed metadata DataFrame
        """
        logger.info("Processing large dataset with chunked loading...")

        with gctx_loader:
            total_rows = (
                gctx_loader._n_rows
                if nrows is None
                else min(nrows, gctx_loader._n_rows)
            )

            # Process in chunks if dataset is large
            if total_rows > chunk_size:
                metadata_chunks = []
                for start in tqdm(
                    range(0, total_rows, chunk_size), desc="Loading metadata"
                ):
                    end = min(start + chunk_size, total_rows)
                    chunk = gctx_loader.get_row_metadata(row_slice=slice(start, end))
                    metadata_chunks.append(chunk)

                metadata = pd.concat(metadata_chunks, ignore_index=True)
            else:
                # Load entire dataset if small enough
                metadata = gctx_loader.get_row_metadata(row_slice=slice(0, total_rows))

        # Ensure metadata is a DataFrame
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        return metadata

    @staticmethod
    def _get_required_columns(
        dataset_type: str, group_by: Optional[str] = None, stratify_by: Optional[str] = None
    ) -> List[str]:
        """Get required columns based on dataset type and splitting parameters."""
        if dataset_type == "multimodal" or dataset_type == "chemical":
            required_cols = ["canonical_smiles", "pert_dose_log2", "viability_clipped"]
        else:  # transcriptomics
            required_cols = ["viability_clipped"]
            
        if group_by:
            required_cols.append(group_by)
        if stratify_by:
            required_cols.append(stratify_by)
            
        return required_cols

    @staticmethod
    def create_and_split_datasets(
        gctx_loader: GCTXDataLoader,
        dataset_type: str = "multimodal",
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = MorganFingerprintTransform(),
        chunk_size: int = 10000,
    ) -> Tuple[T, T, T]:
        """
        Create and split datasets based on dataset type.
        
        Args:
            gctx_loader: GCTX data loader
            dataset_type: Type of dataset to create ('multimodal', 'transcriptomics', or 'chemical')
            feature_space: Gene feature space
            nrows: Number of rows to load
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            group_by: Column to use for group-based splitting
            stratify_by: Column to use for stratified splitting
            transform_transcriptomics: Transform for transcriptomics data
            transform_molecular: Transform for molecular data
            chunk_size: Size of chunks for processing large datasets
            
        Returns:
            Tuple of train, validation, and test datasets
        """
        logger.info(f"Creating and splitting {dataset_type} datasets...")
        
        # Step 1: Load metadata
        metadata = DatasetFactory._load_metadata(gctx_loader, nrows, chunk_size)
        
        # Step 2: Validate required columns
        required_cols = []
        if dataset_type in ["multimodal", "chemical"]:
            required_cols.extend(["canonical_smiles", "pert_dose_log2", "viability_clipped"])
        else:  # transcriptomics
            required_cols.append("viability_clipped")
            
        if group_by:
            required_cols.append(group_by)
        if stratify_by:
            required_cols.append(stratify_by)
            
        DatasetFactory._validate_required_columns(metadata, required_cols)
        
        # Step 3: Split the data
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata=metadata,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            group_by=group_by,
            stratify_by=stratify_by,
        )
        
        # Step 4: Create the datasets based on type
        dataset_args = {}
        
        if dataset_type == "multimodal":
            dataset_class = MultimodalDrugDataset
            dataset_args = {
                "transform_transcriptomics": transform_transcriptomics,
                "transform_molecular": transform_molecular,
            }
        elif dataset_type == "transcriptomics":
            dataset_class = TranscriptomicsDataset
            dataset_args = {
                "transform_transcriptomics": transform_transcriptomics,
            }
        elif dataset_type == "chemical":
            dataset_class = ChemicalDataset
            dataset_args = {
                "transform_molecular": transform_molecular,
            }
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Step 5: Create the datasets
        train_ds, val_ds, test_ds = DatasetFactory._create_dataset_instances(
            gctx_loader=gctx_loader,
            dataset_class=dataset_class,
            dataset_args=dataset_args,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            metadata=metadata,
            feature_space=feature_space,
            dataset_type=dataset_type,
        )
        
        logger.info(f"Created datasets - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds

    @staticmethod
    def _create_dataset_instances(
        gctx_loader: GCTXDataLoader,
        dataset_class: type,
        dataset_args: Dict[str, Any],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        metadata: pd.DataFrame,
        feature_space: Union[str, List[str]],
        dataset_type: str,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create dataset instances for each split.
        
        Args:
            gctx_loader: GCTX data loader
            dataset_class: Dataset class to instantiate
            dataset_args: Arguments for dataset initialization
            train_idx: Training set indices
            val_idx: Validation set indices
            test_idx: Test set indices
            metadata: Metadata DataFrame
            feature_space: Gene feature space
            dataset_type: Type of dataset
            
        Returns:
            Tuple of train, validation, and test datasets
        """
        if dataset_type == "multimodal":
            train_ds = dataset_class(
                transcriptomics_data=gctx_loader.get_expression_data(
                    row_slice=train_idx, feature_space=feature_space
                ),
                metadata=metadata.iloc[train_idx].reset_index(drop=True),
                **dataset_args
            )
            val_ds = dataset_class(
                transcriptomics_data=gctx_loader.get_expression_data(
                    row_slice=val_idx, feature_space=feature_space
                ),
                metadata=metadata.iloc[val_idx].reset_index(drop=True),
                **dataset_args
            )
            test_ds = dataset_class(
                transcriptomics_data=gctx_loader.get_expression_data(
                    row_slice=test_idx, feature_space=feature_space
                ),
                metadata=metadata.iloc[test_idx].reset_index(drop=True),
                **dataset_args
            )
        elif dataset_type == "transcriptomics":
            train_ds = dataset_class(
                transcriptomics_data=gctx_loader.get_expression_data(
                    row_slice=train_idx, feature_space=feature_space
                ),
                viability=metadata.iloc[train_idx]["viability_clipped"].values,
                **dataset_args
            )
            val_ds = dataset_class(
                transcriptomics_data=gctx_loader.get_expression_data(
                    row_slice=val_idx, feature_space=feature_space
                ),
                viability=metadata.iloc[val_idx]["viability_clipped"].values,
                **dataset_args
            )
            test_ds = dataset_class(
                transcriptomics_data=gctx_loader.get_expression_data(
                    row_slice=test_idx, feature_space=feature_space
                ),
                viability=metadata.iloc[test_idx]["viability_clipped"].values,
                **dataset_args
            )
        elif dataset_type == "chemical":
            train_ds = dataset_class(
                smiles=metadata.iloc[train_idx]["canonical_smiles"].values,
                dosage=metadata.iloc[train_idx]["pert_dose_log2"].values,
                viability=metadata.iloc[train_idx]["viability_clipped"].values,
                **dataset_args
            )
            val_ds = dataset_class(
                smiles=metadata.iloc[val_idx]["canonical_smiles"].values,
                dosage=metadata.iloc[val_idx]["pert_dose_log2"].values,
                viability=metadata.iloc[val_idx]["viability_clipped"].values,
                **dataset_args
            )
            test_ds = dataset_class(
                smiles=metadata.iloc[test_idx]["canonical_smiles"].values,
                dosage=metadata.iloc[test_idx]["pert_dose_log2"].values,
                viability=metadata.iloc[test_idx]["viability_clipped"].values,
                **dataset_args
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        return train_ds, val_ds, test_ds

    # Specialized convenience methods
    @staticmethod
    def create_and_split_multimodal(
        gctx_loader: GCTXDataLoader,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform_transcriptomics: Optional[Callable] = None,
        transform_molecular: Optional[Callable] = None,
        chunk_size: int = 10000,
    ) -> Tuple[MultimodalDrugDataset, MultimodalDrugDataset, MultimodalDrugDataset]:
        """Create and split multimodal datasets."""
        return DatasetFactory.create_and_split_datasets(
            gctx_loader=gctx_loader,
            dataset_type="multimodal",
            feature_space=feature_space,
            nrows=nrows,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            group_by=group_by,
            stratify_by=stratify_by,
            transform_transcriptomics=transform_transcriptomics,
            transform_molecular=transform_molecular,
            chunk_size=chunk_size,
        )

    @staticmethod
    def create_and_split_transcriptomics(
        gctx_loader: GCTXDataLoader,
        feature_space: Union[str, List[str]] = "landmark",
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform: Optional[Callable] = None,
        chunk_size: int = 10000,
    ) -> Tuple[TranscriptomicsDataset, TranscriptomicsDataset, TranscriptomicsDataset]:
        """Create and split transcriptomics datasets."""
        return DatasetFactory.create_and_split_datasets(
            gctx_loader=gctx_loader,
            dataset_type="transcriptomics",
            feature_space=feature_space,
            nrows=nrows,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            group_by=group_by,
            stratify_by=stratify_by,
            transform=transform,
            chunk_size=chunk_size,
        )

    @staticmethod
    def create_and_split_chemical(
        gctx_loader: GCTXDataLoader,
        nrows: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        group_by: Optional[str] = None,
        stratify_by: Optional[str] = None,
        transform: Optional[Callable] = None,
        chunk_size: int = 10000,
    ) -> Tuple[ChemicalDataset, ChemicalDataset, ChemicalDataset]:
        """Create and split chemical datasets."""
        return DatasetFactory.create_and_split_datasets(
            gctx_loader=gctx_loader,
            dataset_type="chemical",
            feature_space="landmark", 
            nrows=nrows,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            group_by=group_by,
            stratify_by=stratify_by,
            transform=transform,
            chunk_size=chunk_size,
        )