"""
Main analysis script for psychedelic gradient analysis.

This script orchestrates the complete analysis pipeline from raw fMRI data
to final results and visualizations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.connectivity_analysis import ConnectivityAnalyzer, load_data_from_directory
from analysis.gradient_analysis import GradientAnalyzer, GradientStatistics, create_gradient_dataframe
from analysis.statistical_analysis import ConnectivityStatistics, NetworkAnalysis, create_statistical_report
from visualization.plotting import ConnectivityPlotter, GradientPlotter, StatisticalPlotter, create_summary_figure


class PsychedelicGradientAnalysis:
    """
    Main analysis class for psychedelic gradient analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize analysis with configuration.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or self._default_config()
        self.results = {}
        
        # Initialize analyzers
        self.conn_analyzer = ConnectivityAnalyzer(
            n_rois=self.config['n_rois'],
            standardize=self.config['standardize']
        )
        
        self.grad_analyzer = GradientAnalyzer(
            n_components=self.config['n_gradient_components'],
            random_state=self.config['random_state']
        )
        
        # Initialize plotters
        self.conn_plotter = ConnectivityPlotter()
        self.grad_plotter = GradientPlotter()
        self.stat_plotter = StatisticalPlotter()
        
    def _default_config(self):
        """Default configuration parameters."""
        return {
            'n_rois': 100,
            'n_gradient_components': 10,
            'standardize': True,
            'random_state': 0,
            'alpha': 0.05,
            'correction_method': 'fdr',
            'output_dir': 'results',
            'save_figures': True,
            'figure_format': 'png',
            'figure_dpi': 300
        }
    
    def load_data(self, dmt_data_dir, pcb_data_dir):
        """
        Load fMRI data from directories.
        
        Parameters
        ----------
        dmt_data_dir : str
            Directory containing DMT condition data
        pcb_data_dir : str
            Directory containing placebo condition data
        """
        print("Loading fMRI data...")
        
        # Load file lists
        self.dmt_files = load_data_from_directory(dmt_data_dir, '*.nii')
        self.pcb_files = load_data_from_directory(pcb_data_dir, '*.nii')
        
        print(f"Found {len(self.dmt_files)} DMT files")
        print(f"Found {len(self.pcb_files)} PCB files")
        
        if len(self.dmt_files) != len(self.pcb_files):
            print("Warning: Number of DMT and PCB files don't match!")
        
        # Store in results
        self.results['data_info'] = {
            'n_dmt_files': len(self.dmt_files),
            'n_pcb_files': len(self.pcb_files),
            'dmt_files': self.dmt_files,
            'pcb_files': self.pcb_files
        }
    
    def compute_connectivity(self):
        """Compute functional connectivity matrices."""
        print("Computing functional connectivity matrices...")
        
        # Compute connectivity for both conditions
        print("  Computing DMT connectivity...")
        self.dmt_connectivity = self.conn_analyzer.compute_group_connectivity(self.dmt_files)
        
        print("  Computing PCB connectivity...")
        self.pcb_connectivity = self.conn_analyzer.compute_group_connectivity(self.pcb_files)
        
        # Store results
        self.results['connectivity'] = {
            'dmt': self.dmt_connectivity,
            'pcb': self.pcb_connectivity,
            'labels': self.conn_analyzer.labels
        }
        
        print(f"  Computed connectivity for {len(self.dmt_connectivity)} DMT subjects")
        print(f"  Computed connectivity for {len(self.pcb_connectivity)} PCB subjects")
    
    def compute_gradients(self):
        """Compute functional connectivity gradients."""
        print("Computing functional connectivity gradients...")
        
        # Compute gradients for both conditions
        print("  Computing DMT gradients...")
        self.dmt_gradients_group, self.dmt_gradients_individual = \
            self.grad_analyzer.compute_group_gradients(self.dmt_connectivity)
        
        print("  Computing PCB gradients...")
        self.pcb_gradients_group, self.pcb_gradients_individual = \
            self.grad_analyzer.compute_group_gradients(self.pcb_connectivity)
        
        # Store results
        self.results['gradients'] = {
            'dmt_group': self.dmt_gradients_group,
            'dmt_individual': self.dmt_gradients_individual,
            'pcb_group': self.pcb_gradients_group,
            'pcb_individual': self.pcb_gradients_individual
        }
        
        print(f"  Computed gradients shape: {self.dmt_gradients_group.shape}")
    
    def perform_statistical_tests(self):
        """Perform statistical comparisons between conditions."""
        print("Performing statistical tests...")
        
        # Reshape connectivity data for statistical testing
        dmt_reshaped = ConnectivityStatistics.reshape_connectivity_data(self.dmt_connectivity)
        pcb_reshaped = ConnectivityStatistics.reshape_connectivity_data(self.pcb_connectivity)
        
        # Test normality
        print("  Testing normality assumptions...")
        _, dmt_violations = ConnectivityStatistics.test_normality(dmt_reshaped)
        _, pcb_violations = ConnectivityStatistics.test_normality(pcb_reshaped)
        
        # Paired t-test on connectivity
        print("  Performing paired t-tests on connectivity...")
        t_stats_conn, p_vals_conn = ConnectivityStatistics.paired_t_test_connectivity(
            dmt_reshaped, pcb_reshaped
        )
        
        # Apply significance threshold
        significant_conn = ConnectivityStatistics.apply_significance_threshold(
            t_stats_conn, p_vals_conn, 
            alpha=self.config['alpha'],
            correction=self.config['correction_method']
        )
        
        # Paired t-test on gradients
        print("  Performing paired t-tests on gradients...")
        t_stats_grad, p_vals_grad = GradientStatistics.paired_t_test(
            self.dmt_gradients_individual, self.pcb_gradients_individual
        )
        
        # Compute effect sizes
        print("  Computing effect sizes...")
        effect_sizes_grad = GradientStatistics.compute_effect_sizes(
            self.dmt_gradients_individual, self.pcb_gradients_individual
        )
        
        # Create statistical report
        conn_report = create_statistical_report(
            t_stats_conn, p_vals_conn,
            alpha=self.config['alpha'],
            correction=self.config['correction_method']
        )
        
        # Store results
        self.results['statistics'] = {
            'connectivity': {
                't_statistics': t_stats_conn,
                'p_values': p_vals_conn,
                'significant_mask': significant_conn,
                'report': conn_report
            },
            'gradients': {
                't_statistics': t_stats_grad,
                'p_values': p_vals_grad,
                'effect_sizes': effect_sizes_grad
            },
            'normality': {
                'dmt_violations': len(dmt_violations),
                'pcb_violations': len(pcb_violations),
                'total_tests': np.prod(dmt_reshaped.shape[:2])
            }
        }
        
        print(f"  Found {conn_report['n_significant']} significant connections out of {conn_report['n_total_tests']}")
        print(f"  Normality violations: DMT={len(dmt_violations)}, PCB={len(pcb_violations)}")
    
    def analyze_networks(self):
        """Perform network-level analysis."""
        print("Performing network-level analysis...")
        
        # Get network labels
        network_labels = self.conn_analyzer.get_network_labels()
        
        # Compute network-level connectivity statistics
        dmt_network_stats = NetworkAnalysis.compute_network_statistics(
            self.dmt_connectivity, network_labels, self.conn_analyzer.labels
        )
        
        pcb_network_stats = NetworkAnalysis.compute_network_statistics(
            self.pcb_connectivity, network_labels, self.conn_analyzer.labels
        )
        
        # Compute network differences
        network_diff = dmt_network_stats['mean'] - pcb_network_stats['mean']
        
        # Store results
        self.results['networks'] = {
            'labels': network_labels,
            'dmt_stats': dmt_network_stats,
            'pcb_stats': pcb_network_stats,
            'difference': network_diff
        }
        
        print(f"  Analyzed {len(network_labels)} networks")
    
    def create_visualizations(self):
        """Create all visualizations."""
        if not self.config['save_figures']:
            return
        
        print("Creating visualizations...")
        
        # Create output directory
        fig_dir = Path(self.config['output_dir']) / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot connectivity matrices
        print("  Plotting connectivity matrices...")
        
        # Average connectivity matrices
        dmt_mean_conn = np.mean(self.dmt_connectivity, axis=0)
        pcb_mean_conn = np.mean(self.pcb_connectivity, axis=0)
        
        # DMT connectivity
        self.conn_plotter.plot_connectivity_matrix(
            dmt_mean_conn, 
            title="DMT Mean Connectivity",
            save_path=fig_dir / f"dmt_connectivity.{self.config['figure_format']}"
        )
        
        # PCB connectivity
        self.conn_plotter.plot_connectivity_matrix(
            pcb_mean_conn,
            title="PCB Mean Connectivity", 
            save_path=fig_dir / f"pcb_connectivity.{self.config['figure_format']}"
        )
        
        # Difference matrix
        self.conn_plotter.plot_difference_matrix(
            dmt_mean_conn, pcb_mean_conn,
            title="DMT - PCB Connectivity Difference",
            save_path=fig_dir / f"connectivity_difference.{self.config['figure_format']}"
        )
        
        # Plot gradients
        print("  Plotting gradients...")
        
        # Average gradients
        dmt_mean_grad = np.mean(self.dmt_gradients_individual, axis=0)
        pcb_mean_grad = np.mean(self.pcb_gradients_individual, axis=0)
        
        # Gradient surfaces
        self.grad_plotter.plot_gradients_on_surface(
            dmt_mean_grad, n_components=3,
            titles=['DMT Gradient 1', 'DMT Gradient 2', 'DMT Gradient 3'],
            save_path=fig_dir / f"dmt_gradients.{self.config['figure_format']}"
        )
        
        self.grad_plotter.plot_gradients_on_surface(
            pcb_mean_grad, n_components=3,
            titles=['PCB Gradient 1', 'PCB Gradient 2', 'PCB Gradient 3'],
            save_path=fig_dir / f"pcb_gradients.{self.config['figure_format']}"
        )
        
        # Gradient comparison
        self.grad_plotter.plot_gradient_comparison(
            dmt_mean_grad, pcb_mean_grad, component=0,
            title="Principal Gradient Comparison (DMT vs PCB)",
            save_path=fig_dir / f"gradient_comparison.{self.config['figure_format']}"
        )
        
        # Plot statistical results
        print("  Plotting statistical results...")
        
        self.stat_plotter.plot_t_statistics(
            self.results['statistics']['connectivity']['t_statistics'],
            self.results['statistics']['connectivity']['p_values'],
            alpha=self.config['alpha'],
            title="Connectivity T-Statistics",
            save_path=fig_dir / f"connectivity_ttest.{self.config['figure_format']}"
        )
        
        # Plot network results
        if 'networks' in self.results:
            print("  Plotting network results...")
            
            self.conn_plotter.plot_network_matrix(
                self.results['networks']['difference'],
                list(self.results['networks']['labels'].keys()),
                title="Network-Level Differences (DMT - PCB)",
                save_path=fig_dir / f"network_differences.{self.config['figure_format']}"
            )
        
        # Create summary figure
        print("  Creating summary figure...")
        
        summary_data = {
            'condition1': dmt_mean_conn,
            'condition2': pcb_mean_conn
        }
        
        gradient_data = {
            'condition1': dmt_mean_grad,
            'condition2': pcb_mean_grad
        }
        
        stat_data = {
            't_statistics': self.results['statistics']['connectivity']['t_statistics'],
            'p_values': self.results['statistics']['connectivity']['p_values'],
            'summary': self.results['statistics']['connectivity']['report']
        }
        
        create_summary_figure(
            summary_data, gradient_data, stat_data,
            save_path=fig_dir / f"analysis_summary.{self.config['figure_format']}"
        )
        
        print(f"  All figures saved to: {fig_dir}")
    
    def save_results(self):
        """Save all results to files."""
        print("Saving results...")
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save connectivity matrices
        with open(output_dir / 'connectivity_matrices.pkl', 'wb') as f:
            pickle.dump(self.results['connectivity'], f)
        
        # Save gradients
        with open(output_dir / 'gradients.pkl', 'wb') as f:
            pickle.dump(self.results['gradients'], f)
        
        # Save statistical results
        with open(output_dir / 'statistical_results.pkl', 'wb') as f:
            pickle.dump(self.results['statistics'], f)
        
        # Save network results
        if 'networks' in self.results:
            with open(output_dir / 'network_results.pkl', 'wb') as f:
                pickle.dump(self.results['networks'], f)
        
        # Save summary report
        self._create_summary_report(output_dir / 'analysis_report.txt')
        
        print(f"Results saved to: {output_dir}")
    
    def _create_summary_report(self, output_path):
        """Create a text summary report."""
        with open(output_path, 'w') as f:
            f.write("Psychedelic Gradient Analysis - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Data info
            f.write("Data Information:\n")
            f.write(f"  DMT subjects: {self.results['data_info']['n_dmt_files']}\n")
            f.write(f"  PCB subjects: {self.results['data_info']['n_pcb_files']}\n")
            f.write(f"  ROIs: {self.config['n_rois']}\n")
            f.write(f"  Gradient components: {self.config['n_gradient_components']}\n\n")
            
            # Statistical results
            if 'statistics' in self.results:
                conn_report = self.results['statistics']['connectivity']['report']
                f.write("Statistical Results:\n")
                f.write(f"  Total connectivity tests: {conn_report['n_total_tests']}\n")
                f.write(f"  Significant connections: {conn_report['n_significant']}\n")
                f.write(f"  Proportion significant: {conn_report['proportion_significant']:.4f}\n")
                f.write(f"  Alpha level: {conn_report['alpha_level']}\n")
                f.write(f"  Correction method: {conn_report['correction_method']}\n\n")
                
                # Normality results
                f.write("Normality Test Results:\n")
                norm_results = self.results['statistics']['normality']
                f.write(f"  DMT violations: {norm_results['dmt_violations']}\n")
                f.write(f"  PCB violations: {norm_results['pcb_violations']}\n")
                f.write(f"  Total tests: {norm_results['total_tests']}\n\n")
            
            # Network results
            if 'networks' in self.results:
                f.write("Network Analysis:\n")
                f.write(f"  Number of networks: {len(self.results['networks']['labels'])}\n")
                f.write("  Network names: " + ", ".join(self.results['networks']['labels'].keys()) + "\n\n")
            
            f.write("Analysis completed successfully!\n")
    
    def run_complete_analysis(self, dmt_data_dir, pcb_data_dir):
        """
        Run the complete analysis pipeline.
        
        Parameters
        ----------
        dmt_data_dir : str
            Directory containing DMT condition data
        pcb_data_dir : str
            Directory containing placebo condition data
        """
        print("Starting complete psychedelic gradient analysis...")
        print("=" * 50)
        
        # Run analysis steps
        self.load_data(dmt_data_dir, pcb_data_dir)
        self.compute_connectivity()
        self.compute_gradients()
        self.perform_statistical_tests()
        self.analyze_networks()
        self.create_visualizations()
        self.save_results()
        
        print("=" * 50)
        print("Analysis completed successfully!")
        print(f"Results saved to: {self.config['output_dir']}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='Psychedelic Gradient Analysis')
    parser.add_argument('--dmt-dir', required=True, help='Directory containing DMT fMRI files')
    parser.add_argument('--pcb-dir', required=True, help='Directory containing PCB fMRI files')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--n-rois', type=int, default=100, help='Number of ROIs in atlas')
    parser.add_argument('--n-gradients', type=int, default=10, help='Number of gradient components')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--correction', default='fdr', choices=['fdr', 'bonferroni', 'none'],
                       help='Multiple comparison correction method')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'n_rois': args.n_rois,
        'n_gradient_components': args.n_gradients,
        'alpha': args.alpha,
        'correction_method': args.correction if args.correction != 'none' else None,
        'output_dir': args.output_dir,
        'save_figures': not args.no_figures,
    }
    
    # Run analysis
    analysis = PsychedelicGradientAnalysis(config)
    analysis.run_complete_analysis(args.dmt_dir, args.pcb_dir)


if __name__ == "__main__":
    main()
