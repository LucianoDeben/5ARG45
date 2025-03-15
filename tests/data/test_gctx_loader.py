import pytest
import numpy as np
import pandas as pd
import h5py
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock, mock_open
import time

# Import the GCTXLoader class
# Assuming it's in a module named 'data.loaders'
from src.data.loaders import GCTXLoader

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestGCTXLoader:
    """Test suite for GCTXLoader class."""

    @pytest.fixture
    def mock_h5py_file(self):
        """Create a mock h5py File object with test data structure."""
        mock_file = MagicMock(spec=h5py.File)
        
        # Mock matrix data
        mock_matrix = MagicMock()
        mock_matrix.shape = (100, 200)  # 100 rows, 200 columns
        
        # Create dummy data for matrix slicing
        def mock_getitem(indices):
            if isinstance(indices, tuple) and len(indices) == 2:
                rows, cols = indices
                row_count = 100 if rows == slice(None) else 10
                col_count = 200 if cols == slice(None) else 10
                return np.random.random((row_count, col_count))
            else:
                # Handle single dimension indexing
                count = 10 if isinstance(indices, list) else 100
                mock_result = MagicMock()
                
                # Add a __getitem__ method to handle the second dimension
                def inner_getitem(inner_indices):
                    if isinstance(inner_indices, tuple) and len(inner_indices) == 2:
                        # This is for cases like matrix[rows][:, cols]
                        _, cols = inner_indices
                        col_count = 200 if cols == slice(None) else 10
                        return np.random.random((count, col_count))
                    else:
                        return np.random.random((count, 200))
                
                mock_result.__getitem__ = inner_getitem
                return mock_result
        
        mock_matrix.__getitem__ = mock_getitem
        
        # Set up the mock file structure
        mock_file.__getitem__.side_effect = lambda path: {
            "0/DATA/0/matrix": mock_matrix,
            "0/META/ROW": {
                "id": np.array([f"sample_{i}" for i in range(100)], dtype="S10"),
                "pert_id": np.array([f"pert_{i % 20}" for i in range(100)], dtype="S10"),
                "cell_id": np.array([f"cell_{i % 10}" for i in range(100)], dtype="S10")
            },
            "0/META/COL": {
                "id": np.array([f"gene_{i}" for i in range(200)], dtype="S10"),
                "feature_space": np.array(
                    ["landmark"] * 100 + ["inferred"] * 50 + ["best inferred"] * 50, 
                    dtype="S15"
                )
            },
            "0/META/ROW/id": np.array([f"sample_{i}" for i in range(100)], dtype="S10"),
            "0/META/ROW/pert_id": np.array([f"pert_{i % 20}" for i in range(100)], dtype="S10"),
            "0/META/ROW/cell_id": np.array([f"cell_{i % 10}" for i in range(100)], dtype="S10"),
            "0/META/COL/id": np.array([f"gene_{i}" for i in range(200)], dtype="S10"),
            "0/META/COL/feature_space": np.array(
                ["landmark"] * 100 + ["inferred"] * 50 + ["best inferred"] * 50, 
                dtype="S15"
            )
        }.get(path, MagicMock())
        
        # Mock row and column metadata retrieval
        for meta_type in ["ROW", "COL"]:
            mock_meta = MagicMock()
            keys = ["id", "pert_id", "cell_id"] if meta_type == "ROW" else ["id", "feature_space"]
            mock_meta.keys.return_value = keys
            
            if meta_type == "ROW":
                mock_meta.__getitem__.side_effect = lambda key: {
                    "id": np.array([f"sample_{i}" for i in range(100)], dtype="S10"),
                    "pert_id": np.array([f"pert_{i % 20}" for i in range(100)], dtype="S10"),
                    "cell_id": np.array([f"cell_{i % 10}" for i in range(100)], dtype="S10")
                }.get(key, np.array([]))
            else:
                mock_meta.__getitem__.side_effect = lambda key: {
                    "id": np.array([f"gene_{i}" for i in range(200)], dtype="S10"),
                    "feature_space": np.array(
                        ["landmark"] * 100 + ["inferred"] * 50 + ["best inferred"] * 50, 
                        dtype="S15"
                    )
                }.get(key, np.array([]))
            
            mock_file.__getitem__.return_value = mock_meta

        # Mock file attributes
        mock_file.attrs = {
            "version": b"1.0",
            "source": b"LINCS",
            "created": b"2023-01-01"
        }

        return mock_file

    @pytest.fixture
    def sample_gctx_file(self):
        """Create a temporary test GCTX file."""
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
                
                pert_ids = [f"pert_{i % 20}".encode('utf-8') for i in range(100)]
                row_group.create_dataset("pert_id", data=np.array(pert_ids, dtype='S10'))
                
                cell_ids = [f"cell_{i % 10}".encode('utf-8') for i in range(100)]
                row_group.create_dataset("cell_id", data=np.array(cell_ids, dtype='S10'))
                
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

    def test_init_with_validation(self, sample_gctx_file):
        """Test initialization with validation."""
        loader = GCTXLoader(sample_gctx_file, validate=True)
        assert loader.gctx_file == sample_gctx_file
        assert loader.cache_metadata == True
        assert loader.enable_lru_cache == True
        
    def test_init_without_validation(self, sample_gctx_file):
        """Test initialization without validation."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        assert loader.gctx_file == sample_gctx_file
        
    @patch('h5py.File')
    def test_validate_file_success(self, mock_h5py, mock_h5py_file):
        """Test file validation success."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        mock_h5py_file.__getitem__.side_effect = lambda x: np.zeros((100, 200)) if isinstance(x, tuple) else np.zeros(100)
        
        # Set up the mock file to include the required paths
        mock_h5py_file.__contains__.side_effect = lambda path: path in [
            "0/DATA/0/matrix",
            "0/META/ROW",
            "0/META/COL",
            "0/META/ROW/id",
            "0/META/COL/id"
        ]
        mock_h5py_file["0/DATA/0/matrix"].shape = (100, 100)  # Mock the shape of the matrix
        
        loader = GCTXLoader("dummy.gctx", validate=True)
        # If no exception is raised, validation passed
        assert loader is not None
        mock_h5py_file.__contains__.side_effect = lambda path: path in [
            "0/DATA/0/matrix",
            "0/META/ROW",
            "0/META/COL",
            "0/META/ROW/id",
            "0/META/COL/id"
        ]
        mock_h5py_file["0/DATA/0/matrix"].shape = (100, 100)  # Mock the shape of the matrix
        
        loader = GCTXLoader("dummy.gctx", validate=True)
        # If no exception is raised, validation passed
        assert loader is not None

    @patch('h5py.File')
    def test_validate_file_failure(self, mock_h5py):
        """Test file validation failure."""
        # Make the validation fail by returning a mock that will fail the validation
        mock_file = MagicMock(spec=h5py.File)
        mock_file.__getitem__.side_effect = KeyError("Invalid path")
        mock_h5py.return_value.__enter__.return_value = mock_file
        
        with pytest.raises(ValueError):
            GCTXLoader("invalid.gctx", validate=True)

    @patch('h5py.File')
    def test_dimensions_property(self, mock_h5py, mock_h5py_file):
        """Test dimensions property."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        dims = loader.dimensions
        
        assert dims == (100, 200)
        assert loader.n_rows == 100
        assert loader.n_cols == 200

    @patch('h5py.File')
    def test_get_file_attributes(self, mock_h5py, mock_h5py_file):
        """Test getting file attributes."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        attrs = loader.get_file_attributes()
        
        assert attrs["version"] == "1.0"
        assert attrs["source"] == "LINCS"
        assert attrs["created"] == "2023-01-01"

    @patch('h5py.File')
    def test_get_row_metadata(self, mock_h5py, mock_h5py_file):
        """Test getting row metadata."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        row_meta = loader.get_row_metadata()
        
        assert isinstance(row_meta, pd.DataFrame)
        assert "id" in row_meta.columns
        assert "pert_id" in row_meta.columns
        assert "cell_id" in row_meta.columns
        assert len(row_meta) == 100

    @patch('h5py.File')
    def test_get_row_metadata_with_indices(self, mock_h5py, mock_h5py_file):
        """Test getting row metadata with specific indices."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        row_meta = loader.get_row_metadata(row_indices=[0, 1, 2])
        
        assert isinstance(row_meta, pd.DataFrame)
        assert len(row_meta) == 3
        
    @patch('h5py.File')
    def test_get_row_metadata_with_columns(self, mock_h5py, mock_h5py_file):
        """Test getting row metadata with specific columns."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        row_meta = loader.get_row_metadata(columns=["id"])
        
        assert isinstance(row_meta, pd.DataFrame)
        assert list(row_meta.columns) == ["id"]
        
    @patch('h5py.File')
    def test_get_column_metadata(self, mock_h5py, mock_h5py_file):
        """Test getting column metadata."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        col_meta = loader.get_column_metadata()
        
        assert isinstance(col_meta, pd.DataFrame)
        assert "id" in col_meta.columns
        assert "feature_space" in col_meta.columns
        assert len(col_meta) == 200

    @patch('h5py.File')
    def test_get_column_metadata_with_feature_space(self, mock_h5py, mock_h5py_file):
        """Test getting column metadata filtered by feature space."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        col_meta = loader.get_column_metadata(feature_space="landmark")
        
        assert isinstance(col_meta, pd.DataFrame)
        assert len(col_meta) == 100  # 100 landmark genes in our mock

    @patch('h5py.File')
    def test_get_row_ids(self, mock_h5py, mock_h5py_file):
        """Test getting row IDs."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        row_ids = loader.get_row_ids()
        
        assert isinstance(row_ids, list)
        assert len(row_ids) == 100
        assert row_ids[0] == "sample_0"

    @patch('h5py.File')
    def test_get_gene_symbols(self, mock_h5py, mock_h5py_file):
        """Test getting gene symbols."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        gene_symbols = loader.get_gene_symbols()
        
        assert isinstance(gene_symbols, list)
        assert len(gene_symbols) == 200
        assert gene_symbols[0] == "gene_0"

    @patch('h5py.File')
    def test_get_data_matrix(self, mock_h5py, mock_h5py_file):
        """Test getting data matrix."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        data = loader.get_data_matrix()
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 200)
        
    @patch('h5py.File')
    def test_get_data_matrix_with_indices(self, mock_h5py, mock_h5py_file):
        """Test getting data matrix with specific indices."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        data = loader.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)
        
    @patch('h5py.File')
    def test_get_data_matrix_with_feature_space(self, mock_h5py, mock_h5py_file):
        """Test getting data matrix filtered by feature space."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        data = loader.get_data_matrix(feature_space="landmark")
        
        assert isinstance(data, np.ndarray)
        assert data.shape[1] == 100  # 100 landmark genes in our mock

    def test_get_data_matrix_real_file(self, sample_gctx_file):
        """Test getting data matrix from a real file."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        data = loader.get_data_matrix()
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 200)
        
    def test_get_data_matrix_with_indices_real_file(self, sample_gctx_file):
        """Test getting data matrix with specific indices from a real file."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        data = loader.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)
    
    def test_get_data_matrix_with_slices_real_file(self, sample_gctx_file):
        """Test getting data matrix with slices from a real file."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        data = loader.get_data_matrix(row_indices=slice(0, 10), col_indices=slice(0, 15))
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (10, 15)

    @patch('h5py.File')
    def test_get_complete_dataset(self, mock_h5py, mock_h5py_file):
        """Test getting complete dataset (data + metadata)."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        data, row_meta, col_meta = loader.get_complete_dataset()
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 200)
        assert isinstance(row_meta, pd.DataFrame)
        assert len(row_meta) == 100
        assert isinstance(col_meta, pd.DataFrame)
        assert len(col_meta) == 200

    def test_cache_management(self, sample_gctx_file):
        """Test cache management and statistics."""
        loader = GCTXLoader(
            sample_gctx_file, 
            cache_metadata=True, 
            enable_lru_cache=True,
            max_matrix_cache_size=2
        )
        
        # Initially no cache hits
        stats = loader.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Make some data requests to test caching
        data1 = loader.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        data2 = loader.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        
        # Should have 1 miss and 1 hit now
        stats = loader.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        
        # Clear cache
        loader.clear_cache()
        
        # Stats should be reset
        stats = loader.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
    def test_cache_eviction(self, sample_gctx_file):
        """Test LRU cache eviction policy."""
        # Set max cache size very small to force eviction
        loader = GCTXLoader(
            sample_gctx_file, 
            cache_metadata=True, 
            enable_lru_cache=True,
            max_matrix_cache_size=2
        )
        
        # Make 3 different data requests to test eviction (with max_size=2)
        loader.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        loader.get_data_matrix(row_indices=[3, 4, 5], col_indices=[8, 9, 10])
        loader.get_data_matrix(row_indices=[6, 7, 8], col_indices=[11, 12, 13])
        
        # Should have cache count of 2 (max size)
        stats = loader.get_cache_stats()
        assert stats["matrices_cached"] == 2
        assert stats["max_matrices"] == 2
        
        # The first request should be evicted, so this should be a miss
        loader.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        
        # Misses should be 2 now (initial miss + after eviction)
        stats = loader.get_cache_stats()
        assert stats["misses"] == 4  # 3 initial loads + 1 after eviction
        
    def test_context_manager(self, sample_gctx_file):
        """Test context manager usage."""
        with GCTXLoader(sample_gctx_file, validate=False) as loader:
            # File should be open during the context
            assert loader._file_handle is not None
            
            # Should be able to access data
            data = loader.get_data_matrix(row_indices=[0, 1, 2])
            assert data is not None
            
        # File should be closed after context
        assert loader._file_handle is None

    @patch('h5py.File')
    def test_handle_unordered_indices(self, mock_h5py, mock_h5py_file):
        """Test handling of unordered indices."""
        mock_h5py.return_value.__enter__.return_value = mock_h5py_file
        
        loader = GCTXLoader("dummy.gctx", validate=False)
        
        # Test with ordered indices
        indices = [0, 1, 2, 3]
        sorted_indices, restore_order = loader._handle_unordered_indices(indices)
        assert np.array_equal(sorted_indices, indices)
        assert restore_order is None
        
        # Test with unordered indices
        indices = [3, 1, 0, 2]
        sorted_indices, restore_order = loader._handle_unordered_indices(indices)
        assert np.array_equal(sorted_indices, [0, 1, 2, 3])
        assert restore_order is not None
        
        # Check restore_order works
        ordered_data = np.array([10, 20, 30, 40])  # Data after loading with sorted indices
        expected_original_order = np.array([30, 20, 10, 40])  # Expected data in original order
        
        restored_data = ordered_data[restore_order]
        assert np.array_equal(restored_data, expected_original_order)
        
    def test_error_handling_invalid_indices(self, sample_gctx_file):
        """Test error handling for invalid indices."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        
        # Test with out-of-bounds index
        with pytest.raises(IndexError):
            loader.get_data_matrix(row_indices=[1000])
            
        # Test with invalid index type
        with pytest.raises(TypeError):
            loader.get_data_matrix(row_indices="invalid")
            
    def test_error_handling_conflicting_params(self, sample_gctx_file):
        """Test error handling for conflicting parameters."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        
        # Test with both col_indices and feature_space
        with pytest.raises(ValueError):
            loader.get_data_matrix(col_indices=[0, 1], feature_space="landmark")
            
    def test_parallel_loading(self, sample_gctx_file):
        """Test parallel data loading."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        
        # Create indices that would trigger parallel loading
        row_indices = list(range(50))
        
        # Test parallel loading with multiple workers
        data = loader.get_data_matrix_parallel(
            row_indices=row_indices, 
            chunk_size=10,
            n_workers=2
        )
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (50, 200)
        
    def test_performance_caching(self, sample_gctx_file):
        """Test performance improvement with caching."""
        # Create loader with caching enabled
        loader_cached = GCTXLoader(
            sample_gctx_file, 
            cache_metadata=True, 
            enable_lru_cache=True
        )
        
        # Create loader with caching disabled
        loader_uncached = GCTXLoader(
            sample_gctx_file, 
            cache_metadata=False, 
            enable_lru_cache=False
        )
        
        # Measure time for repeated cached access
        start_time = time.time()
        for _ in range(5):
            loader_cached.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        cached_time = time.time() - start_time
        
        # Measure time for repeated uncached access
        start_time = time.time()
        for _ in range(5):
            loader_uncached.get_data_matrix(row_indices=[0, 1, 2], col_indices=[5, 6, 7])
        uncached_time = time.time() - start_time
        
        # Cached should be significantly faster (allowing some tolerance for test variability)
        # Note: This test may be flaky on some systems or CI environments, so we use a generous ratio
        assert cached_time < uncached_time * 0.9, "Caching should improve performance"
        
    @pytest.mark.parametrize("feature_space", [
        "landmark", 
        "inferred", 
        "best inferred", 
        ["landmark", "inferred"],
        "all"
    ])
    def test_feature_space_filtering(self, sample_gctx_file, feature_space):
        """Test feature space filtering with various options."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        
        # Get data with feature space filtering
        data = loader.get_data_matrix(feature_space=feature_space)
        
        # Verify the correct number of columns based on the feature space
        if feature_space == "landmark":
            assert data.shape[1] == 100
        elif feature_space == "inferred":
            assert data.shape[1] == 50
        elif feature_space == "best inferred":
            assert data.shape[1] == 50
        elif feature_space == ["landmark", "inferred"]:
            assert data.shape[1] == 150
        elif feature_space == "all":
            assert data.shape[1] == 200
            
    def test_invalid_feature_space(self, sample_gctx_file):
        """Test error handling for invalid feature space."""
        loader = GCTXLoader(sample_gctx_file, validate=False)
        
        # Test with invalid feature space
        with pytest.raises(ValueError):
            loader.get_data_matrix(feature_space="invalid")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])