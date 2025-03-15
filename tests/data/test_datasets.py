import os
import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import pickle
import h5py
from unittest.mock import patch, MagicMock, mock_open

# Import the datasets module and components to test
from src.data.datasets import (
    MultimodalDrugDataset,
    TranscriptomicsDataset,
    MolecularDataset,
    DatasetFactory
)
from src.data.loaders import GCTXLoader
from src.data.feature_transforms import MorganFingerprintTransform


class TestDrugDatasets:
    """Test class for the dataset classes."""

    @pytest.fixture
    def sample_transcriptomics_data(self):
        """Create sample transcriptomics data."""
        return np.random.random((10, 100))  # 10 samples, 100 genes

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return pd.DataFrame({
            "viability_clipped": np.random.random(10),
            "canonical_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"] * 10,  # Aspirin SMILES string
            "pert_dose_log2": np.random.random(10),
            "cell_id": ["A375"] * 5 + ["PC3"] * 5,
            "pert_id": [f"BRD-{i}" for i in range(10)]
        })

    @pytest.fixture
    def sample_row_ids(self):
        """Create sample row ids."""
        return [f"sample_{i}" for i in range(10)]

    @pytest.fixture
    def sample_gene_symbols(self):
        """Create sample gene symbols."""
        return [f"gene_{i}" for i in range(100)]

    @pytest.fixture
    def mock_gctx_loader(self):
        """Create a mock GCTXLoader."""
        mock_loader = MagicMock(spec=GCTXLoader)
        mock_loader.gctx_file = "test.gctx"
        mock_loader.n_rows = 100
        mock_loader.n_cols = 200
        
        # Mock the get_data_matrix method
        mock_loader.get_data_matrix.return_value = np.random.random((10, 100))
        
        # Mock the get_row_metadata method
        mock_loader.get_row_metadata.return_value = pd.DataFrame({
            "id": [f"sample_{i}" for i in range(100)],
            "cell_id": ["A375"] * 50 + ["PC3"] * 50,
            "pert_id": [f"BRD-{i % 20}" for i in range(100)],
            "canonical_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"] * 100,
            "pert_dose_log2": np.random.random(100),
            "viability_clipped": np.random.random(100)
        })
        
        # Mock the get_row_ids method
        mock_loader.get_row_ids.return_value = [f"sample_{i}" for i in range(10)]
        
        # Mock the get_gene_symbols method
        mock_loader.get_gene_symbols.return_value = [f"gene_{i}" for i in range(100)]
        
        # Mock dimensions
        mock_loader.dimensions = (100, 200)
        
        return mock_loader

    @pytest.fixture
    def mock_transform_transcriptomics(self):
        """Create a mock transform for transcriptomics data."""
        def transform(x):
            return x * 2
        return transform

    @pytest.fixture
    def mock_transform_molecular(self):
        """Create a mock transform for molecular data."""
        def transform(x):
            # Return a random fingerprint for the SMILES
            fingerprint = np.random.randint(0, 2, 128)
            return torch.tensor(fingerprint, dtype=torch.float32)
        return transform

    def test_multimodal_drug_dataset_init(self, sample_transcriptomics_data, sample_metadata, 
                                          sample_row_ids, sample_gene_symbols):
        """Test initialization of MultimodalDrugDataset."""
        dataset = MultimodalDrugDataset(
            transcriptomics_data=sample_transcriptomics_data,
            metadata=sample_metadata,
            row_ids=sample_row_ids,
            gene_symbols=sample_gene_symbols
        )
        
        assert len(dataset) == 10
        assert dataset.transcriptomics_data.shape == (10, 100)
        assert len(dataset.viability) == 10
        assert len(dataset.smiles) == 10
        assert len(dataset.dosage) == 10
        assert len(dataset.row_ids) == 10
        assert len(dataset.gene_symbols) == 100

    def test_multimodal_drug_dataset_getitem(self, sample_transcriptomics_data, sample_metadata, 
                                             sample_row_ids, sample_gene_symbols,
                                             mock_transform_transcriptomics, mock_transform_molecular):
        """Test __getitem__ method of MultimodalDrugDataset."""
        dataset = MultimodalDrugDataset(
            transcriptomics_data=sample_transcriptomics_data,
            metadata=sample_metadata,
            row_ids=sample_row_ids,
            gene_symbols=sample_gene_symbols,
            transform_transcriptomics=mock_transform_transcriptomics,
            transform_molecular=mock_transform_molecular
        )
        
        # Get first item
        item = dataset[0]
        
        # Check keys
        assert "transcriptomics" in item
        assert "molecular" in item
        assert "dosage" in item
        assert "viability" in item
        assert "row_id" in item
        assert "gene_symbols" in item
        
        # Check types
        assert isinstance(item["transcriptomics"], torch.Tensor)
        assert isinstance(item["molecular"], torch.Tensor)
        assert isinstance(item["dosage"], torch.Tensor)
        assert isinstance(item["viability"], torch.Tensor)
        assert isinstance(item["row_id"], str)
        assert isinstance(item["gene_symbols"], list)
        
        # Check shapes
        assert item["transcriptomics"].shape == (100,)  # Number of genes
        assert item["molecular"].shape == (128,)  # Fingerprint length
        assert item["dosage"].shape == ()  # Scalar
        assert item["viability"].shape == ()  # Scalar
        
        # Check transform effects
        assert torch.allclose(item["transcriptomics"], 
                              torch.tensor(sample_transcriptomics_data[0] * 2, dtype=torch.float32))

    def test_transcriptomics_dataset(self, sample_transcriptomics_data, sample_metadata, 
                                     sample_row_ids, sample_gene_symbols, 
                                     mock_transform_transcriptomics):
        """Test TranscriptomicsDataset."""
        dataset = TranscriptomicsDataset(
            transcriptomics_data=sample_transcriptomics_data,
            viability=sample_metadata["viability_clipped"].values,
            row_ids=sample_row_ids,
            gene_symbols=sample_gene_symbols,
            transform_transcriptomics=mock_transform_transcriptomics
        )
        
        assert len(dataset) == 10
        
        # Test getitem
        item = dataset[0]
        assert "transcriptomics" in item
        assert "viability" in item
        assert "row_id" in item
        assert "gene_symbols" in item
        
        # Check transform effects
        assert torch.allclose(item["transcriptomics"], 
                              torch.tensor(sample_transcriptomics_data[0] * 2, dtype=torch.float32))

    def test_molecular_dataset(self, sample_metadata, sample_row_ids, 
                               mock_transform_molecular):
        """Test MolecularDataset."""
        dataset = MolecularDataset(
            smiles=sample_metadata["canonical_smiles"].values,
            dosage=sample_metadata["pert_dose_log2"].values,
            viability=sample_metadata["viability_clipped"].values,
            row_ids=sample_row_ids,
            transform_molecular=mock_transform_molecular
        )
        
        assert len(dataset) == 10
        
        # Test getitem
        item = dataset[0]
        assert "molecular" in item
        assert "dosage" in item
        assert "viability" in item
        assert "row_id" in item
        
        # Check types
        assert isinstance(item["molecular"], torch.Tensor)
        assert isinstance(item["dosage"], torch.Tensor)
        assert isinstance(item["viability"], torch.Tensor)
        assert isinstance(item["row_id"], str)
        
        # Check shapes
        assert item["molecular"].shape == (128,)  # Fingerprint length
        assert item["dosage"].shape == ()  # Scalar
        assert item["viability"].shape == ()  # Scalar


class TestDatasetFactory:
    """Test class for the DatasetFactory."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for all tests."""
        # Use a temporary directory for cache
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original cache dir
            original_cache_dir = DatasetFactory.CACHE_DIR
            # Set cache dir to temp directory
            DatasetFactory.CACHE_DIR = temp_dir
            
            # Reset cache statistics
            DatasetFactory._memory_cache = {}
            DatasetFactory._memory_cache_timestamps = {}
            DatasetFactory._memory_cache_hits = 0
            DatasetFactory._memory_cache_misses = 0
            
            yield
            
            # Restore original cache dir
            DatasetFactory.CACHE_DIR = original_cache_dir

    @pytest.fixture
    def sample_gctx_file(self):
        """Create a sample GCTX file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.gctx', delete=False) as tmp:
            # Create a sample GCTX file
            with h5py.File(tmp.name, 'w') as f:
                # Create matrix dataset
                matrix = f.create_dataset("0/DATA/0/matrix", (100, 200), dtype='float32')
                
                # Fill with random data
                matrix[:] = np.random.random((100, 200)).astype('float32')
                
                # Create row metadata
                row_group = f.create_group("0/META/ROW")
                row_ids = [f"sample_{i}".encode('utf-8') for i in range(100)]
                row_group.create_dataset("id", data=np.array(row_ids, dtype='S10'))
                
                # Add cell_id, pert_id
                cell_ids = ["A375".encode('utf-8')] * 50 + ["PC3".encode('utf-8')] * 50
                row_group.create_dataset("cell_id", data=np.array(cell_ids, dtype='S10'))
                
                pert_ids = [f"BRD-{i % 20}".encode('utf-8') for i in range(100)]
                row_group.create_dataset("pert_id", data=np.array(pert_ids, dtype='S10'))
                
                # Add smiles and dosage
                smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O".encode('utf-8')] * 100
                row_group.create_dataset("canonical_smiles", data=np.array(smiles, dtype='S40'))
                
                dosage = np.random.random(100)
                row_group.create_dataset("pert_dose_log2", data=dosage)
                
                # Add viability
                viability = np.random.random(100)
                row_group.create_dataset("viability_clipped", data=viability)
                
                # Create column metadata
                col_group = f.create_group("0/META/COL")
                gene_ids = [f"gene_{i}".encode('utf-8') for i in range(200)]
                col_group.create_dataset("id", data=np.array(gene_ids, dtype='S10'))
                
                feature_space = (["landmark"] * 100 + ["inferred"] * 50 + ["best inferred"] * 50)
                col_group.create_dataset("feature_space", 
                                         data=np.array([fs.encode('utf-8') for fs in feature_space], 
                                                      dtype='S15'))
                
                # Set file attributes
                f.attrs["version"] = b"1.0"
                f.attrs["source"] = b"LINCS"
                f.attrs["created"] = b"2023-01-01"
        
        yield tmp.name
        # Clean up the file
        os.unlink(tmp.name)

    @pytest.fixture
    def mock_morgan_transform(self):
        """Create a mock Morgan fingerprint transform."""
        transform = MagicMock(spec=MorganFingerprintTransform)
        transform.return_value = np.ones(128, dtype=np.float32)
        return transform

    def test_cache_key_generation(self, mock_gctx_loader):
        """Test cache key generation."""
        key1 = DatasetFactory._get_cache_key(
            mock_gctx_loader.gctx_file, "multimodal", "landmark", None, 42, None, None
        )
        key2 = DatasetFactory._get_cache_key(
            mock_gctx_loader.gctx_file, "multimodal", "landmark", None, 42, None, None
        )
        key3 = DatasetFactory._get_cache_key(
            mock_gctx_loader.gctx_file, "transcriptomics", "landmark", None, 42, None, None
        )
        
        # Same parameters should generate same key
        assert key1 == key2
        # Different parameters should generate different keys
        assert key1 != key3

    def test_cache_operations(self):
        """Test cache save and load operations."""
        # Generate test data
        test_data = ("train", "val", "test")
        cache_key = "test_key"
        
        # Test save to cache
        result = DatasetFactory._save_to_cache(cache_key, test_data)
        assert result is True
        
        # Test load from cache
        loaded_data = DatasetFactory._load_from_cache(cache_key)
        assert loaded_data == test_data
        
        # Clear cache
        DatasetFactory.clear_cache(memory_cache=True, disk_cache=True)
        
        # Check cache is cleared
        loaded_data = DatasetFactory._load_from_cache(cache_key)
        assert loaded_data is None

    def test_validate_split_sizes(self):
        """Test validation of split sizes."""
        # Valid split sizes
        DatasetFactory._validate_split_sizes(0.2, 0.1)
        
        # Invalid split sizes
        with pytest.raises(ValueError):
            DatasetFactory._validate_split_sizes(0, 0.1)
        
        with pytest.raises(ValueError):
            DatasetFactory._validate_split_sizes(0.2, 0)
        
        with pytest.raises(ValueError):
            DatasetFactory._validate_split_sizes(0.5, 0.5)  # Sum equals 1
        
        with pytest.raises(ValueError):
            DatasetFactory._validate_split_sizes(0.7, 0.4)  # Sum exceeds 1

    def test_validate_required_columns(self):
        """Test validation of required columns."""
        # Create test metadata
        metadata = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        # Valid column requirements
        DatasetFactory._validate_required_columns(metadata, ["col1", "col2"])
        
        # Invalid column requirements
        with pytest.raises(ValueError):
            DatasetFactory._validate_required_columns(metadata, ["col1", "col3"])

    def test_split_data(self):
        """Test data splitting functionality."""
        # Create test metadata
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        metadata = pd.DataFrame({
            "id": range(n_samples),
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 40,
            "strat": np.concatenate([np.random.random(50) * 0.5, np.random.random(50) * 0.5 + 0.5])
        })
        
        # Test standard splitting
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata=metadata,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        # Check sizes
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples
        
        # Check no overlap
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0
        
        # Test group-based splitting
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata=metadata,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            group_by="group"
        )
        
        # Check sizes
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples
        
        # Check no overlap
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0
        
        # Test stratified splitting
        train_idx, val_idx, test_idx = DatasetFactory._split_data(
            metadata=metadata,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            stratify_by="strat"
        )
        
        # Check sizes
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples
        
        # Check no overlap
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0
        
        # Check stratification
        train_strat = metadata.iloc[train_idx]["strat"].mean()
        val_strat = metadata.iloc[val_idx]["strat"].mean()
        test_strat = metadata.iloc[test_idx]["strat"].mean()
        
        # Means should be somewhat similar if stratified correctly
        assert abs(train_strat - val_strat) < 0.2
        assert abs(train_strat - test_strat) < 0.2

    @patch("src.data.datasets.DatasetFactory._add_to_memory_cache")
    @patch("src.data.datasets.DatasetFactory._save_to_cache")
    def test_create_and_split_datasets(self, mock_save_cache, mock_add_memory_cache, sample_gctx_file, mock_morgan_transform):
        """Test the create_and_split_datasets method."""
        # Create a real GCTXLoader with the sample file
        loader = GCTXLoader(sample_gctx_file, validate=True, cache_metadata=True)
        
        # Test multimodal dataset creation
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_datasets(
            gctx_loader=loader,
            dataset_type="multimodal",
            feature_space="landmark",
            nrows=None,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            transform_molecular=mock_morgan_transform,
            use_cache=True
        )
        
        # Check types
        assert isinstance(train_ds, MultimodalDrugDataset)
        assert isinstance(val_ds, MultimodalDrugDataset)
        assert isinstance(test_ds, MultimodalDrugDataset)
        
        # Check that caching was used
        assert mock_add_memory_cache.called
        assert mock_save_cache.called
        
        # Check all data was used
        assert len(train_ds) + len(val_ds) + len(test_ds) == loader.n_rows
        
        # Test one item from train dataset
        item = train_ds[0]
        assert "transcriptomics" in item
        assert "molecular" in item
        assert "dosage" in item
        assert "viability" in item
        assert "row_id" in item
        assert "gene_symbols" in item

    @patch("src.data.datasets.DatasetFactory._add_to_memory_cache")
    @patch("src.data.datasets.DatasetFactory._save_to_cache")
    def test_create_and_split_transcriptomics(self, mock_save_cache, mock_add_memory_cache, sample_gctx_file):
        """Test the create_and_split_transcriptomics method."""
        # Create a real GCTXLoader with the sample file
        loader = GCTXLoader(sample_gctx_file, validate=True, cache_metadata=True)
        
        # Test transcriptomics dataset creation
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
            gctx_loader=loader,
            feature_space="landmark",
            nrows=None,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            use_cache=True
        )
        
        # Check types
        assert isinstance(train_ds, TranscriptomicsDataset)
        assert isinstance(val_ds, TranscriptomicsDataset)
        assert isinstance(test_ds, TranscriptomicsDataset)
        
        # Check that caching was used
        assert mock_add_memory_cache.called
        assert mock_save_cache.called
        
        # Check all data was used
        assert len(train_ds) + len(val_ds) + len(test_ds) == loader.n_rows
        
        # Test one item from train dataset
        item = train_ds[0]
        assert "transcriptomics" in item
        assert "viability" in item
        assert "row_id" in item
        assert "gene_symbols" in item

    @patch("src.data.datasets.DatasetFactory._add_to_memory_cache")
    @patch("src.data.datasets.DatasetFactory._save_to_cache")
    def test_create_and_split_chemical(self, mock_save_cache, mock_add_memory_cache, sample_gctx_file, mock_morgan_transform):
        """Test the create_and_split_chemical method."""
        # Create a real GCTXLoader with the sample file
        loader = GCTXLoader(sample_gctx_file, validate=True, cache_metadata=True)
        
        # Test chemical dataset creation
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_chemical(
            gctx_loader=loader,
            nrows=None,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            transform_molecular=mock_morgan_transform,
            use_cache=True
        )
        
        # Check types
        assert isinstance(train_ds, MolecularDataset)
        assert isinstance(val_ds, MolecularDataset)
        assert isinstance(test_ds, MolecularDataset)
        
        # Check that caching was used
        assert mock_add_memory_cache.called
        assert mock_save_cache.called
        
        # Check all data was used
        assert len(train_ds) + len(val_ds) + len(test_ds) == loader.n_rows
        
        # Test one item from train dataset
        item = train_ds[0]
        assert "molecular" in item
        assert "dosage" in item
        assert "viability" in item
        assert "row_id" in item

    def test_cache_stats(self, sample_gctx_file, mock_morgan_transform):
        """Test the cache stats functionality."""
        # Create a real GCTXLoader with the sample file
        loader = GCTXLoader(sample_gctx_file, validate=True, cache_metadata=True)
        
        # Clear cache first
        DatasetFactory.clear_cache(memory_cache=True, disk_cache=True)
        
        # Get initial stats
        initial_stats = DatasetFactory.get_cache_stats()
        assert initial_stats["memory_cache_hits"] == 0
        assert initial_stats["memory_cache_misses"] == 0
        
        # Create datasets once (should be a cache miss)
        DatasetFactory.create_and_split_multimodal(
            gctx_loader=loader,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            transform_molecular=mock_morgan_transform,
            use_cache=True
        )
        
        # Check stats after first creation
        after_miss_stats = DatasetFactory.get_cache_stats()
        assert after_miss_stats["memory_cache_hits"] == 0
        assert after_miss_stats["memory_cache_misses"] == 1
        
        # Create datasets again with same parameters (should be a cache hit)
        DatasetFactory.create_and_split_multimodal(
            gctx_loader=loader,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            transform_molecular=mock_morgan_transform,
            use_cache=True
        )
        
        # Check stats after cache hit
        after_hit_stats = DatasetFactory.get_cache_stats()
        assert after_hit_stats["memory_cache_hits"] == 1
        assert after_hit_stats["memory_cache_misses"] == 1

    def test_memory_cache_eviction(self, sample_gctx_file, mock_morgan_transform):
        """Test the memory cache eviction policy."""
        # Create a real GCTXLoader with the sample file
        loader = GCTXLoader(sample_gctx_file, validate=True, cache_metadata=True)
        
        # Set max cache size very small to force eviction
        original_size = DatasetFactory._max_memory_cache_size
        DatasetFactory._max_memory_cache_size = 2
        
        # Clear cache first
        DatasetFactory.clear_cache(memory_cache=True, disk_cache=True)
        
        # Create datasets with different parameters to fill cache
        for i in range(3):  # Create 3 different cached items (exceeding max of 2)
            DatasetFactory.create_and_split_multimodal(
                gctx_loader=loader,
                test_size=0.2,
                val_size=0.1,
                random_state=42 + i,  # Different random state for each
                transform_molecular=mock_morgan_transform,
                use_cache=True
            )
        
        # Check that cache size is limited to max
        stats = DatasetFactory.get_cache_stats()
        assert stats["memory_cache_items"] <= DatasetFactory._max_memory_cache_size
        
        # Restore original max size
        DatasetFactory._max_memory_cache_size = original_size

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_cache_error_handling(self, mock_pickle_dump, mock_file_open, sample_gctx_file):
        """Test error handling in cache operations."""
        # Make pickle.dump raise an exception
        mock_pickle_dump.side_effect = Exception("Mocked error")
        
        # Attempt to save to cache
        result = DatasetFactory._save_to_cache("test_key", "test_data")
        assert result is False  # Should return False on error
        
        # Make file open raise an exception
        mock_file_open.side_effect = Exception("Mocked error")
        
        # Attempt to load from cache
        data = DatasetFactory._load_from_cache("test_key")
        assert data is None  # Should return None on error


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])