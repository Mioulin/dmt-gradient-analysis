"""
Statistical analysis functions for connectivity and gradient data.

This module provides comprehensive statistical testing functions for
comparing brain connectivity and gradients between conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, ttest_rel, ttest_ind
import warnings


class ConnectivityStatistics:
    """
    Statistical analysis class for connectivity data.
    """
    
    @staticmethod
    def reshape_connectivity_data(connectivity_list):
        """
        Reshape connectivity matrices for statistical analysis.
        
        Parameters
        ----------
        connectivity_list : list
            List of connectivity matrices (n_subjects, n_rois, n_rois)
            
        Returns
        -------
        reshaped_data : np.ndarray
            Data reshaped to (n_rois, n_rois, n_subjects)
        """
        n_subjects = len(connectivity_list)
        n_rois = connectivity_list[0].shape[0]
        
        reshaped_data = np.empty([n_rois, n_rois, n_subjects])
        
        for i in range(n_rois):
            for j in range(n_rois):
                for k in range(n_subjects):
                    reshaped_data[i, j, k] = connectivity_list[k][i, j]
        
        return reshaped_data
    
    @staticmethod
    def test_normality(data, p_threshold=0.05):
        """
        Test normality using Shapiro-Wilk test.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test for normality
        p_threshold : float, default=0.05
            P-value threshold for normality
            
        Returns
        -------
        is_normal : bool
            Whether data is normally distributed
        p_value : float
            P-value from Shapiro-Wilk test
        violations : list
            List of indices where normality is violated
        """
        violations = []
        
        if data.ndim == 3:  # (n_rois, n_rois, n_subjects)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    _, p_value = shapiro(data[i, j, :])
                    if p_value < p_threshold:
                        violations.append((i, j, p_value))
        
        elif data.ndim == 2:  # (n_features, n_subjects)
            for i in range(data.shape[0]):
                _, p_value = shapiro(data[i, :])
                if p_value < p_threshold:
                    violations.append((i, p_value))
        
        else:  # 1D data
            _, p_value = shapiro(data)
            if p_value < p_threshold:
                violations.append(p_value)
        
        is_normal = len(violations) == 0
        return is_normal, violations
    
    @staticmethod
    def paired_t_test_connectivity(conn_data1, conn_data2):
        """
        Perform paired t-test on connectivity matrices.
        
        Parameters
        ----------
        conn_data1, conn_data2 : np.ndarray
            Connectivity data shaped as (n_rois, n_rois, n_subjects)
            
        Returns
        -------
        t_statistics : np.ndarray
            T-statistics matrix (n_rois, n_rois)
        p_values : np.ndarray
            P-values matrix (n_rois, n_rois)
        significant_mask : np.ndarray
            Boolean mask of significant connections
        """
        n_rois = conn_data1.shape[0]
        t_statistics = np.empty([n_rois, n_rois])
        p_values = np.empty([n_rois, n_rois])
        
        for i in range(n_rois):
            for j in range(n_rois):
                t_stat, p_val = ttest_rel(conn_data1[i, j, :], conn_data2[i, j, :])
                t_statistics[i, j] = t_stat
                p_values[i, j] = p_val
        
        return t_statistics, p_values
    
    @staticmethod
    def create_significance_mask(p_values, alpha=0.05, correction=None):
        """
        Create significance mask with optional multiple comparison correction.
        
        Parameters
        ----------
        p_values : np.ndarray
            Matrix of p-values
        alpha : float, default=0.05
            Significance level
        correction : str, optional
            Multiple comparison correction ('bonferroni', 'fdr')
            
        Returns
        -------
        significant_mask : np.ndarray
            Boolean mask of significant values
        corrected_alpha : float
            Corrected alpha level (if correction applied)
        """
        if correction == 'bonferroni':
            n_tests = np.prod(p_values.shape)
            corrected_alpha = alpha / n_tests
            significant_mask = p_values < corrected_alpha
        
        elif correction == 'fdr':
            from statsmodels.stats.multitest import fdrcorrection
            # Flatten p-values for FDR correction
            p_flat = p_values.flatten()
            rejected, p_corrected = fdrcorrection(p_flat, alpha=alpha)
            significant_mask = rejected.reshape(p_values.shape)
            corrected_alpha = np.max(p_flat[rejected]) if np.any(rejected) else 0
        
        else:
            significant_mask = p_values < alpha
            corrected_alpha = alpha
        
        return significant_mask, corrected_alpha
    
    @staticmethod
    def apply_significance_threshold(data_matrix, p_values, alpha=0.05, correction=None):
        """
        Apply significance threshold to data matrix.
        
        Parameters
        ----------
        data_matrix : np.ndarray
            Data matrix (e.g., t-statistics or differences)
        p_values : np.ndarray
            Corresponding p-values
        alpha : float, default=0.05
            Significance level
        correction : str, optional
            Multiple comparison correction
            
        Returns
        -------
        thresholded_matrix : np.ndarray
            Data matrix with non-significant values set to zero
        """
        significant_mask, _ = ConnectivityStatistics.create_significance_mask(
            p_values, alpha, correction
        )
        
        thresholded_matrix = data_matrix.copy()
        thresholded_matrix[~significant_mask] = 0
        
        return thresholded_matrix


class NetworkAnalysis:
    """
    Network-level analysis functions.
    """
    
    @staticmethod
    def aggregate_by_networks(data_matrix, network_labels, roi_labels, 
                             aggregation_func=np.mean):
        """
        Aggregate connectivity data by brain networks.
        
        Parameters
        ----------
        data_matrix : np.ndarray
            Data matrix to aggregate (n_rois, n_rois)
        network_labels : dict
            Dictionary mapping network names to ROI indices
        roi_labels : list
            List of ROI labels
        aggregation_func : callable, default=np.mean
            Function to use for aggregation
            
        Returns
        -------
        network_matrix : np.ndarray
            Network-level aggregated matrix
        network_names : list
            Names of the networks
        """
        # Map ROI labels to network indices
        roi_to_network = {}
        for network_name, roi_indices in network_labels.items():
            for roi_idx in roi_indices:
                roi_to_network[roi_idx] = network_name
        
        # Get unique network names
        network_names = list(network_labels.keys())
        n_networks = len(network_names)
        
        # Initialize network matrix
        network_matrix = np.zeros((n_networks, n_networks))
        
        # Aggregate data
        for i, net1 in enumerate(network_names):
            for j, net2 in enumerate(network_names):
                roi_indices1 = network_labels[net1]
                roi_indices2 = network_labels[net2]
                
                # Extract submatrix
                submatrix = data_matrix[np.ix_(roi_indices1, roi_indices2)]
                
                # Aggregate
                network_matrix[i, j] = aggregation_func(submatrix)
        
        return network_matrix, network_names
    
    @staticmethod
    def compute_network_statistics(connectivity_matrices, network_labels, roi_labels):
        """
        Compute network-level statistics across subjects.
        
        Parameters
        ----------
        connectivity_matrices : list
            List of connectivity matrices
        network_labels : dict
            Dictionary mapping network names to ROI indices
        roi_labels : list
            List of ROI labels
            
        Returns
        -------
        network_stats : dict
            Dictionary with network statistics
        """
        network_names = list(network_labels.keys())
        n_subjects = len(connectivity_matrices)
        n_networks = len(network_names)
        
        # Initialize arrays for network connectivity values
        network_connectivity = np.zeros((n_subjects, n_networks, n_networks))
        
        # Aggregate each subject's data by networks
        for subj_idx, conn_matrix in enumerate(connectivity_matrices):
            network_matrix, _ = NetworkAnalysis.aggregate_by_networks(
                conn_matrix, network_labels, roi_labels
            )
            network_connectivity[subj_idx] = network_matrix
        
        # Compute statistics
        network_stats = {
            'mean': np.mean(network_connectivity, axis=0),
            'std': np.std(network_connectivity, axis=0),
            'connectivity_matrices': network_connectivity,
            'network_names': network_names
        }
        
        return network_stats


class EffectSizeCalculator:
    """
    Effect size calculation functions.
    """
    
    @staticmethod
    def cohens_d(group1, group2, axis=0):
        """
        Calculate Cohen's d effect size.
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Data arrays for the two groups
        axis : int, default=0
            Axis along which to compute effect size
            
        Returns
        -------
        effect_size : np.ndarray
            Cohen's d effect sizes
        """
        mean1 = np.mean(group1, axis=axis)
        mean2 = np.mean(group2, axis=axis)
        
        std1 = np.std(group1, axis=axis, ddof=1)
        std2 = np.std(group2, axis=axis, ddof=1)
        
        n1 = group1.shape[axis]
        n2 = group2.shape[axis]
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        effect_size = (mean1 - mean2) / pooled_std
        
        return effect_size
    
    @staticmethod
    def hedges_g(group1, group2, axis=0):
        """
        Calculate Hedges' g effect size (bias-corrected Cohen's d).
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Data arrays for the two groups
        axis : int, default=0
            Axis along which to compute effect size
            
        Returns
        -------
        effect_size : np.ndarray
            Hedges' g effect sizes
        """
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2, axis)
        
        n1 = group1.shape[axis]
        n2 = group2.shape[axis]
        
        # Bias correction factor
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        
        hedges_g = cohens_d * correction_factor
        
        return hedges_g


class MultipleComparisons:
    """
    Multiple comparison correction methods.
    """
    
    @staticmethod
    def bonferroni_correction(p_values, alpha=0.05):
        """
        Apply Bonferroni correction.
        
        Parameters
        ----------
        p_values : np.ndarray
            Array of p-values
        alpha : float, default=0.05
            Family-wise error rate
            
        Returns
        -------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        corrected_alpha : float
            Corrected alpha level
        """
        n_tests = np.prod(p_values.shape)
        corrected_alpha = alpha / n_tests
        rejected = p_values < corrected_alpha
        
        return rejected, corrected_alpha
    
    @staticmethod
    def fdr_correction(p_values, alpha=0.05, method='indep'):
        """
        Apply False Discovery Rate correction.
        
        Parameters
        ----------
        p_values : np.ndarray
            Array of p-values
        alpha : float, default=0.05
            False discovery rate
        method : str, default='indep'
            Method for FDR correction
            
        Returns
        -------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        corrected_p : np.ndarray
            Corrected p-values
        """
        from statsmodels.stats.multitest import fdrcorrection
        
        original_shape = p_values.shape
        p_flat = p_values.flatten()
        
        rejected_flat, corrected_p_flat = fdrcorrection(p_flat, alpha=alpha, method=method)
        
        rejected = rejected_flat.reshape(original_shape)
        corrected_p = corrected_p_flat.reshape(original_shape)
        
        return rejected, corrected_p


def create_statistical_report(t_stats, p_values, effect_sizes=None, 
                            alpha=0.05, correction=None):
    """
    Create a comprehensive statistical report.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics matrix
    p_values : np.ndarray
        P-values matrix
    effect_sizes : np.ndarray, optional
        Effect sizes matrix
    alpha : float, default=0.05
        Significance level
    correction : str, optional
        Multiple comparison correction method
        
    Returns
    -------
    report : dict
        Statistical report dictionary
    """
    # Apply multiple comparison correction if specified
    if correction:
        if correction == 'bonferroni':
            significant_mask, corrected_alpha = MultipleComparisons.bonferroni_correction(
                p_values, alpha
            )
        elif correction == 'fdr':
            significant_mask, corrected_p = MultipleComparisons.fdr_correction(
                p_values, alpha
            )
            corrected_alpha = alpha
        else:
            raise ValueError(f"Unknown correction method: {correction}")
    else:
        significant_mask = p_values < alpha
        corrected_alpha = alpha
    
    # Count significant connections
    n_total = np.prod(p_values.shape)
    n_significant = np.sum(significant_mask)
    
    # Summary statistics
    report = {
        'n_total_tests': n_total,
        'n_significant': n_significant,
        'proportion_significant': n_significant / n_total,
        'alpha_level': alpha,
        'corrected_alpha': corrected_alpha,
        'correction_method': correction,
        't_statistics': {
            'mean': np.mean(t_stats),
            'std': np.std(t_stats),
            'min': np.min(t_stats),
            'max': np.max(t_stats)
        },
        'p_values': {
            'mean': np.mean(p_values),
            'median': np.median(p_values),
            'min': np.min(p_values),
            'max': np.max(p_values)
        },
        'significant_mask': significant_mask
    }
    
    if effect_sizes is not None:
        report['effect_sizes'] = {
            'mean': np.mean(effect_sizes),
            'std': np.std(effect_sizes),
            'min': np.min(effect_sizes),
            'max': np.max(effect_sizes),
            'mean_significant': np.mean(effect_sizes[significant_mask]) if n_significant > 0 else np.nan
        }
    
    return report


def save_statistical_results(results, output_path):
    """
    Save statistical results to file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing statistical results
    output_path : str
        Path to save the results
    """
    import pickle
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Statistical results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Statistical analysis module loaded successfully!")
    
    # Create example data
    n_subjects = 20
    n_rois = 100
    
    # Simulate connectivity data
    np.random.seed(42)
    conn_data1 = np.random.randn(n_rois, n_rois, n_subjects)
    conn_data2 = np.random.randn(n_rois, n_rois, n_subjects) + 0.2  # Small effect
    
    # Make symmetric
    for i in range(n_subjects):
        conn_data1[:, :, i] = (conn_data1[:, :, i] + conn_data1[:, :, i].T) / 2
        conn_data2[:, :, i] = (conn_data2[:, :, i] + conn_data2[:, :, i].T) / 2
    
    # Perform statistical tests
    t_stats, p_vals = ConnectivityStatistics.paired_t_test_connectivity(
        conn_data1, conn_data2
    )
    
    # Test normality
    is_normal, violations = ConnectivityStatistics.test_normality(conn_data1)
    
    # Create report
    report = create_statistical_report(t_stats, p_vals, correction='fdr')
    
    print(f"Number of significant connections: {report['n_significant']}")
    print(f"Proportion significant: {report['proportion_significant']:.3f}")
    print(f"Normality violations: {len(violations)}")

        