"""
Gradient analysis for functional connectivity data.

This module provides functions for computing and analyzing functional connectivity 
gradients using the BrainSpace toolbox.
"""

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_parcellation, load_conte69
import matplotlib.pyplot as plt


class GradientAnalyzer:
    """
    A class for computing and analyzing functional connectivity gradients.
    
    Parameters
    ----------
    n_components : int, default=10
        Number of gradient components to compute
    approach : str, default='pca'
        Dimensionality reduction approach ('pca', 'dm', 'le')
    kernel : str, default='normalized_angle'
        Kernel for affinity matrix computation
    random_state : int, default=0
        Random state for reproducibility
    """
    
    def __init__(self, n_components=10, approach='pca', kernel='normalized_angle', random_state=0):
        self.n_components = n_components
        self.approach = approach
        self.kernel = kernel
        self.random_state = random_state
        
        # Initialize gradient maps object
        self.gm = GradientMaps(
            n_components=n_components,
            approach=approach,
            kernel=kernel,
            random_state=random_state
        )
        
    def compute_gradients(self, connectivity_matrix):
        """
        Compute gradients from a connectivity matrix.
        
        Parameters
        ----------
        connectivity_matrix : np.ndarray
            Connectivity matrix of shape (n_rois, n_rois)
            
        Returns
        -------
        gradients : np.ndarray
            Gradient values of shape (n_rois, n_components)
        """
        # Fit the gradient maps
        self.gm.fit(connectivity_matrix)
        
        return self.gm.gradients_
    
    def compute_group_gradients(self, connectivity_matrices):
        """
        Compute gradients for a group of connectivity matrices.
        
        Parameters
        ----------
        connectivity_matrices : list
            List of connectivity matrices
            
        Returns
        -------
        group_gradients : np.ndarray
            Gradients for all subjects, shape (n_rois, n_subjects * n_components)
        individual_gradients : list
            List of individual gradient arrays
        """
        individual_gradients = []
        
        for conn_matrix in connectivity_matrices:
            gradients = self.compute_gradients(conn_matrix)
            individual_gradients.append(gradients)
        
        # Concatenate all gradients
        group_gradients = np.concatenate(individual_gradients, axis=1)
        
        return group_gradients, individual_gradients
    
    def align_gradients(self, target_gradients, source_gradients):
        """
        Align gradients using Procrustes analysis.
        
        Parameters
        ----------
        target_gradients : np.ndarray
            Target gradients for alignment
        source_gradients : np.ndarray
            Source gradients to be aligned
            
        Returns
        -------
        aligned_gradients : np.ndarray
            Aligned source gradients
        """
        from brainspace.gradient.alignment import procrustes_alignment
        
        aligned = procrustes_alignment(source_gradients, target_gradients)
        return aligned
    
    def compute_gradient_similarity(self, gradients1, gradients2, method='correlation'):
        """
        Compute similarity between two sets of gradients.
        
        Parameters
        ----------
        gradients1, gradients2 : np.ndarray
            Gradient arrays to compare
        method : str, default='correlation'
            Similarity method ('correlation', 'cosine')
            
        Returns
        -------
        similarity : float or np.ndarray
            Similarity measure(s)
        """
        if method == 'correlation':
            if gradients1.ndim == 1 and gradients2.ndim == 1:
                return np.corrcoef(gradients1, gradients2)[0, 1]
            else:
                # Compute correlation for each component
                similarities = []
                for i in range(min(gradients1.shape[1], gradients2.shape[1])):
                    corr = np.corrcoef(gradients1[:, i], gradients2[:, i])[0, 1]
                    similarities.append(corr)
                return np.array(similarities)
        
        elif method == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(gradients1.T, gradients2.T)
        
        else:
            raise ValueError(f"Method {method} not supported")


class GradientStatistics:
    """
    Statistical analysis functions for gradients.
    """
    
    @staticmethod
    def paired_t_test(gradients_condition1, gradients_condition2):
        """
        Perform paired t-test between two conditions.
        
        Parameters
        ----------
        gradients_condition1, gradients_condition2 : list
            Lists of gradient arrays for each condition
            
        Returns
        -------
        t_stats : np.ndarray
            T-statistics for each ROI and component
        p_values : np.ndarray
            P-values for each ROI and component
        """
        from scipy.stats import ttest_rel
        
        # Convert to arrays if needed
        grad1 = np.array(gradients_condition1)  # (n_subjects, n_rois, n_components)
        grad2 = np.array(gradients_condition2)
        
        # Perform t-test for each ROI and component
        t_stats = np.zeros((grad1.shape[1], grad1.shape[2]))
        p_values = np.zeros((grad1.shape[1], grad1.shape[2]))
        
        for roi in range(grad1.shape[1]):
            for comp in range(grad1.shape[2]):
                t_stat, p_val = ttest_rel(grad1[:, roi, comp], grad2[:, roi, comp])
                t_stats[roi, comp] = t_stat
                p_values[roi, comp] = p_val
        
        return t_stats, p_values
    
    @staticmethod
    def compute_effect_sizes(gradients_condition1, gradients_condition2):
        """
        Compute Cohen's d effect sizes.
        
        Parameters
        ----------
        gradients_condition1, gradients_condition2 : list
            Lists of gradient arrays for each condition
            
        Returns
        -------
        effect_sizes : np.ndarray
            Cohen's d effect sizes
        """
        grad1 = np.array(gradients_condition1)
        grad2 = np.array(gradients_condition2)
        
        # Compute means and pooled standard deviation
        mean1 = np.mean(grad1, axis=0)
        mean2 = np.mean(grad2, axis=0)
        
        std1 = np.std(grad1, axis=0, ddof=1)
        std2 = np.std(grad2, axis=0, ddof=1)
        
        n1, n2 = grad1.shape[0], grad2.shape[0]
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        effect_sizes = (mean1 - mean2) / pooled_std
        return effect_sizes


def create_gradient_dataframe(gradients, labels, subject_ids=None, condition_labels=None):
    """
    Create a pandas DataFrame from gradient data.
    
    Parameters
    ----------
    gradients : np.ndarray or list
        Gradient data
    labels : list
        ROI labels
    subject_ids : list, optional
        Subject identifiers
    condition_labels : list, optional
        Condition labels
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with gradient data
    """
    if isinstance(gradients, list):
        # Multiple subjects
        data_list = []
        for i, grad in enumerate(gradients):
            for comp in range(grad.shape[1]):
                for roi, label in enumerate(labels):
                    row = {
                        'subject_id': subject_ids[i] if subject_ids else i,
                        'roi_label': label.decode('utf-8') if isinstance(label, bytes) else label,
                        'roi_index': roi,
                        'component': comp,
                        'gradient_value': grad[roi, comp],
                    }
                    if condition_labels:
                        row['condition'] = condition_labels[i]
                    data_list.append(row)
        
        df = pd.DataFrame(data_list)
    else:
        # Single subject
        data_list = []
        for comp in range(gradients.shape[1]):
            for roi, label in enumerate(labels):
                row = {
                    'roi_label': label.decode('utf-8') if isinstance(label, bytes) else label,
                    'roi_index': roi,
                    'component': comp,
                    'gradient_value': gradients[roi, comp],
                }
                data_list.append(row)
        
        df = pd.DataFrame(data_list)
    
    return df


if __name__ == "__main__":
    # Example usage
    analyzer = GradientAnalyzer(n_components=3)
    
    # Create example connectivity matrix
    n_rois = 100
    example_conn = np.random.rand(n_rois, n_rois)
    example_conn = (example_conn + example_conn.T) / 2  # Make symmetric
    np.fill_diagonal(example_conn, 1)  # Set diagonal to 1
    
    # Compute gradients
    gradients = analyzer.compute_gradients(example_conn)
    
    print("GradientAnalyzer initialized successfully!")
    print(f"Computed gradients shape: {gradients.shape}")
    print(f"Number of components: {analyzer.n_components}")
