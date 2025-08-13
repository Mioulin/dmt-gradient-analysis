"""
Functional connectivity analysis for psychedelic neuroimaging data.

This module provides functions for computing functional connectivity matrices
from fMRI time series data using various parcellations and connectivity measures.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import warnings
warnings.filterwarnings('ignore')


class ConnectivityAnalyzer:
    """
    A class for analyzing functional connectivity from fMRI data.
    
    Parameters
    ----------
    atlas_name : str, default='schaefer'
        Name of the atlas to use for parcellation
    n_rois : int, default=100
        Number of ROIs in the parcellation
    standardize : bool, default=True
        Whether to standardize time series
    """
    
    def __init__(self, atlas_name='schaefer', n_rois=100, standardize=True):
        self.atlas_name = atlas_name
        self.n_rois = n_rois
        self.standardize = standardize
        self.atlas_file = None
        self.labels = None
        self.masker = None
        self._setup_atlas()
    
    def _setup_atlas(self):
        """Setup the brain atlas and masker."""
        if self.atlas_name == 'schaefer':
            atlas_data = datasets.fetch_atlas_schaefer_2018(
                n_rois=self.n_rois, 
                yeo_networks=7,
                resolution_mm=2
            )
            self.atlas_file = atlas_data.maps
            self.labels = atlas_data.labels
        else:
            raise ValueError(f"Atlas {self.atlas_name} not supported")
        
        # Setup masker
        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas_file,
            standardize=self.standardize,
            memory='nilearn_cache',
            verbose=0
        )
    
    def compute_connectivity_matrix(self, nifti_file, kind='correlation'):
        """
        Compute connectivity matrix from a NIfTI file.
        
        Parameters
        ----------
        nifti_file : str
            Path to the NIfTI file
        kind : str, default='correlation'
            Type of connectivity measure ('correlation', 'covariance', etc.)
            
        Returns
        -------
        connectivity_matrix : np.ndarray
            Connectivity matrix of shape (n_rois, n_rois)
        """
        # Extract time series
        time_series = self.masker.fit_transform(nifti_file)
        
        # Compute connectivity
        correlation_measure = ConnectivityMeasure(kind=kind)
        connectivity_matrix = correlation_measure.fit_transform([time_series])[0]
        
        return connectivity_matrix
    
    def compute_group_connectivity(self, nifti_files, kind='correlation'):
        """
        Compute connectivity matrices for a group of subjects.
        
        Parameters
        ----------
        nifti_files : list
            List of paths to NIfTI files
        kind : str, default='correlation'
            Type of connectivity measure
            
        Returns
        -------
        connectivity_matrices : list
            List of connectivity matrices
        """
        connectivity_matrices = []
        
        for nifti_file in nifti_files:
            conn_matrix = self.compute_connectivity_matrix(nifti_file, kind=kind)
            connectivity_matrices.append(conn_matrix)
        
        return connectivity_matrices
    
    def get_network_labels(self):
        """
        Get network labels for ROIs.
        
        Returns
        -------
        network_labels : dict
            Dictionary mapping network names to ROI indices
        """
        if self.atlas_name == 'schaefer':
            # Parse Schaefer labels to extract network information
            network_labels = {}
            for i, label in enumerate(self.labels):
                label_str = label.decode('utf-8') if isinstance(label, bytes) else label
                # Extract network name (e.g., 'LH_Vis_1' -> 'LH_Vis')
                parts = label_str.split('_')
                if len(parts) >= 3:
                    network = '_'.join(parts[:2])  # e.g., 'LH_Vis'
                    if network not in network_labels:
                        network_labels[network] = []
                    network_labels[network].append(i)
            
            return network_labels
        else:
            raise ValueError(f"Network extraction not implemented for {self.atlas_name}")


def load_data_from_directory(data_dir, file_pattern='*.nii'):
    """
    Load NIfTI files from a directory.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing NIfTI files
    file_pattern : str, default='*.nii'
        File pattern to match
        
    Returns
    -------
    file_list : list
        List of NIfTI file paths
    """
    import os
    import glob
    
    pattern = os.path.join(data_dir, file_pattern)
    file_list = glob.glob(pattern)
    file_list.sort()  # Ensure consistent ordering
    
    return file_list


def save_connectivity_matrices(connectivity_matrices, output_path, labels=None):
    """
    Save connectivity matrices to file.
    
    Parameters
    ----------
    connectivity_matrices : list or np.ndarray
        Connectivity matrices to save
    output_path : str
        Path to save the matrices
    labels : list, optional
        ROI labels
    """
    import pickle
    
    data = {
        'connectivity_matrices': connectivity_matrices,
        'labels': labels,
        'n_subjects': len(connectivity_matrices) if isinstance(connectivity_matrices, list) else 1
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # Example usage
    analyzer = ConnectivityAnalyzer(n_rois=100)
    
    # Example: load and analyze data
    # dmt_files = load_data_from_directory('/path/to/dmt/data')
    # pcb_files = load_data_from_directory('/path/to/pcb/data')
    
    # dmt_matrices = analyzer.compute_group_connectivity(dmt_files)
    # pcb_matrices = analyzer.compute_group_connectivity(pcb_files)
    
    print("ConnectivityAnalyzer initialized successfully!")
    print(f"Atlas: {analyzer.atlas_name}")
    print(f"Number of ROIs: {analyzer.n_rois}")
    print(f"Number of labels: {len(analyzer.labels)}")
