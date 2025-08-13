"""
Tests for connectivity analysis module.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append('../src')

from analysis.connectivity_analysis import ConnectivityAnalyzer


class TestConnectivityAnalyzer:
    """Test class for ConnectivityAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ConnectivityAnalyzer(n_rois=100)
        
    def test_initialization(self):
        """Test proper initialization of ConnectivityAnalyzer."""
        assert self.analyzer.n_rois == 100
        assert self.analyzer.atlas_name == 'schaefer'
        assert self.analyzer.standardize is True
        assert self.analyzer.masker is not None
        assert len(self.analyzer.labels) == 100
    
    def test_invalid_atlas(self):
        """Test that invalid atlas raises ValueError."""
        with pytest.raises(ValueError):
            ConnectivityAnalyzer(atlas_name='invalid_atlas')
    
    @patch('analysis.connectivity_analysis.datasets.fetch_atlas_schaefer_2018')
    def test_atlas_setup(self, mock_fetch):
        """Test atlas setup with mocked data."""
        # Mock atlas data
        mock_atlas = Mock()
        mock_atlas.maps = "fake_atlas_path"
        mock_atlas.labels = [f"ROI_{i}".encode() for i in range(100)]
        mock_fetch.return_value = mock_atlas
        
        analyzer = ConnectivityAnalyzer(n_rois=100)
        
        mock_fetch.assert_called_once_with(
            n_rois=100, 
            yeo_networks=7, 
            resolution_mm=2
        )
        assert analyzer.atlas_file == "fake_atlas_path"
        assert len(analyzer.labels) == 100
    
    def test_compute_connectivity_matrix_shape(self):
        """Test that connectivity matrix has correct shape."""
        # Create mock time series data
        n_timepoints = 200
        n_rois = 100
        mock_timeseries = np.random.randn(n_timepoints, n_rois)
        
        # Mock the masker
        with patch.object(self.analyzer.masker, 'fit_transform', return_value=mock_timeseries):
            conn_matrix = self.analyzer.compute_connectivity_matrix('fake_file.nii')
            
            assert conn_matrix.shape == (n_rois, n_rois)
            assert np.allclose(conn_matrix, conn_matrix.T)  # Should be symmetric
            assert np.allclose(np.diag(conn_matrix), 1.0)   # Diagonal should be 1
    
    def test_compute_group_connectivity(self):
        """Test group connectivity computation."""
        n_subjects = 5
        n_rois = 100
        fake_files = [f'subject_{i}.nii' for i in range(n_subjects)]
        
        # Mock individual connectivity computation
        mock_matrix = np.random.rand(n_rois, n_rois)
        mock_matrix = (mock_matrix + mock_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(mock_matrix, 1)
        
        with patch.object(self.analyzer, 'compute_connectivity_matrix', 
                         return_value=mock_matrix):
            matrices = self.analyzer.compute_group_connectivity(fake_files)
            
            assert len(matrices) == n_subjects
            assert all(matrix.shape == (n_rois, n_rois) for matrix in matrices)
    
    def test_get_network_labels(self):
        """Test network label extraction."""
        network_labels = self.analyzer.get_network_labels()
        
        assert isinstance(network_labels, dict)
        assert len(network_labels) > 0
        
        # Check that all ROIs are assigned to networks
        total_rois = sum(len(rois) for rois in network_labels.values())
        assert total_rois == self.analyzer.n_rois
        
        # Check that network names follow expected pattern
        for network_name in network_labels.keys():
            assert '_' in network_name  # Should contain underscore
    
    def test_connectivity_matrix_properties(self):
        """Test properties of computed connectivity matrices."""
        n_timepoints = 200
        n_rois = 100
        
        # Create realistic time series with some correlation structure
        np.random.seed(42)
        base_signal = np.random.randn(n_timepoints, 1)
        noise = np.random.randn(n_timepoints, n_rois) * 0.5
        mock_timeseries = base_signal + noise  # All ROIs correlated with base signal
        
        with patch.object(self.analyzer.masker, 'fit_transform', return_value=mock_timeseries):
            conn_matrix = self.analyzer.compute_connectivity_matrix('fake_file.nii')
            
            # Test matrix properties
            assert np.allclose(conn_matrix, conn_matrix.T, atol=1e-10)  # Symmetric
            assert np.all(np.diag(conn_matrix) == 1.0)  # Diagonal is 1
            assert np.all(conn_matrix >= -1) and np.all(conn_matrix <= 1)  # Valid correlation range
    
    def test_different_connectivity_kinds(self):
        """Test different types of connectivity measures."""
        n_timepoints = 200
        n_rois = 100
        mock_timeseries = np.random.randn(n_timepoints, n_rois)
        
        connectivity_kinds = ['correlation', 'covariance']
        
        for kind in connectivity_kinds:
            with patch.object(self.analyzer.masker, 'fit_transform', return_value=mock_timeseries):
                conn_matrix = self.analyzer.compute_connectivity_matrix('fake_file.nii', kind=kind)
                
                assert conn_matrix.shape == (n_rois, n_rois)
                assert np.allclose(conn_matrix, conn_matrix.T)


class TestUtilityFunctions:
    """Test utility functions in connectivity analysis module."""
    
    def test_load_data_from_directory(self):
        """Test loading data files from directory."""
        from analysis.connectivity_analysis import load_data_from_directory
        
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test .nii files
            test_files = ['test1.nii', 'test2.nii', 'test3.nii', 'other.txt']
            for filename in test_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write('fake data')
            
            # Test loading .nii files
            nii_files = load_data_from_directory(temp_dir, '*.nii')
            
            assert len(nii_files) == 3  # Should find 3 .nii files
            assert all(fname.endswith('.nii') for fname in nii_files)
            assert all(os.path.exists(fname) for fname in nii_files)
    
    def test_save_connectivity_matrices(self):
        """Test saving connectivity matrices."""
        from analysis.connectivity_analysis import save_connectivity_matrices
        import pickle
        
        # Create test data
        n_subjects = 3
        n_rois = 50
        test_matrices = [np.random.rand(n_rois, n_rois) for _ in range(n_subjects)]
        test_labels = [f'ROI_{i}' for i in range(n_rois)]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save matrices
            save_connectivity_matrices(test_matrices, temp_path, test_labels)
            
            # Load and verify
            with open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            assert 'connectivity_matrices' in loaded_data
            assert 'labels' in loaded_data
            assert 'n_subjects' in loaded_data
            
            assert len(loaded_data['connectivity_matrices']) == n_subjects
            assert loaded_data['labels'] == test_labels
            assert loaded_data['n_subjects'] == n_subjects
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.fixture
def sample_connectivity_data():
    """Fixture providing sample connectivity data for tests."""
    n_subjects = 5
    n_rois = 50
    np.random.seed(42)
    
    matrices = []
    for _ in range(n_subjects):
        matrix = np.random.rand(n_rois, n_rois)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 1)  # Set diagonal to 1
        matrices.append(matrix)
    
    labels = [f'ROI_{i}'.encode() for i in range(n_rois)]
    
    return {
        'matrices': matrices,
        'labels': labels,
        'n_subjects': n_subjects,
        'n_rois': n_rois
    }


def test_connectivity_analyzer_with_sample_data(sample_connectivity_data):
    """Integration test with sample data."""
    data = sample_connectivity_data
    
    # Test that we can work with the sample data format
    assert len(data['matrices']) == data['n_subjects']
    assert all(matrix.shape == (data['n_rois'], data['n_rois']) 
              for matrix in data['matrices'])
    
    # Test network label extraction would work
    analyzer = ConnectivityAnalyzer(n_rois=data['n_rois'])
    
    # Mock the labels to match our sample data format
    analyzer.labels = data['labels']
    
    network_labels = analyzer.get_network_labels()
    assert isinstance(network_labels, dict)


if __name__ == "__main__":
    pytest.main([__file__])
