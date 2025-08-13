"""
Visualization functions for connectivity and gradient analysis.

This module provides comprehensive plotting functions for brain connectivity
matrices, gradients, and statistical results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')


class ConnectivityPlotter:
    """
    Class for plotting connectivity matrices and related visualizations.
    """
    
    def __init__(self, figsize=(10, 8), style='whitegrid'):
        """
        Initialize plotter with default settings.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Default figure size
        style : str, default='whitegrid'
            Seaborn style
        """
        self.figsize = figsize
        sns.set_style(style)
    
    def plot_connectivity_matrix(self, matrix, labels=None, title=None, 
                                vmin=-1, vmax=1, cmap='RdBu_r', 
                                figsize=None, save_path=None):
        """
        Plot a connectivity matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Connectivity matrix to plot
        labels : list, optional
            ROI labels
        title : str, optional
            Plot title
        vmin, vmax : float
            Color scale limits
        cmap : str, default='RdBu_r'
            Colormap
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        if figsize is None:
            figsize = self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use nilearn plotting if available, otherwise use matplotlib
        try:
            from nilearn import plotting
            plotting.plot_matrix(
                matrix, 
                labels=labels,
                vmax=vmax, 
                vmin=vmin,
                figure=fig,
                axes=ax,
                colorbar=True,
                reorder=True
            )
        except:
            # Fallback to matplotlib
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            if labels is not None:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_network_matrix(self, matrix, network_names, title=None,
                           vmin=None, vmax=None, cmap='RdBu_r',
                           figsize=None, save_path=None):
        """
        Plot a network-level connectivity matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Network connectivity matrix
        network_names : list
            Network names
        title : str, optional
            Plot title
        vmin, vmax : float, optional
            Color scale limits
        cmap : str, default='RdBu_r'
            Colormap
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        if figsize is None:
            figsize = (8, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set default color limits if not provided
        if vmin is None:
            vmin = np.min(matrix)
        if vmax is None:
            vmax = np.max(matrix)
        
        # Plot matrix
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Connectivity Strength', fontsize=12)
        
        # Set labels
        ax.set_xticks(range(len(network_names)))
        ax.set_yticks(range(len(network_names)))
        ax.set_xticklabels(network_names, rotation=45, ha='right')
        ax.set_yticklabels(network_names)
        
        # Add values as text
        for i in range(len(network_names)):
            for j in range(len(network_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_difference_matrix(self, matrix1, matrix2, labels=None, 
                              title="Difference Matrix", vmin=None, vmax=None,
                              cmap='RdBu_r', figsize=None, save_path=None):
        """
        Plot the difference between two connectivity matrices.
        
        Parameters
        ----------
        matrix1, matrix2 : np.ndarray
            Connectivity matrices to compare
        labels : list, optional
            ROI labels
        title : str, default="Difference Matrix"
            Plot title
        vmin, vmax : float, optional
            Color scale limits
        cmap : str, default='RdBu_r'
            Colormap
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        difference = matrix1 - matrix2
        
        # Set symmetric color limits if not provided
        if vmin is None and vmax is None:
            abs_max = np.max(np.abs(difference))
            vmin, vmax = -abs_max, abs_max
        
        return self.plot_connectivity_matrix(
            difference, labels=labels, title=title,
            vmin=vmin, vmax=vmax, cmap=cmap,
            figsize=figsize, save_path=save_path
        )


class GradientPlotter:
    """
    Class for plotting gradient-related visualizations.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_gradients_on_surface(self, gradients, surface_data=None, 
                                 n_components=3, titles=None, 
                                 cmap='viridis', save_path=None):
        """
        Plot gradients on brain surface.
        
        Parameters
        ----------
        gradients : np.ndarray
            Gradient data (n_rois, n_components)
        surface_data : dict, optional
            Surface mesh data
        n_components : int, default=3
            Number of gradient components to plot
        titles : list, optional
            Titles for each component
        cmap : str, default='viridis'
            Colormap
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib figure
            Figure object
        """
        n_components = min(n_components, gradients.shape[1])
        
        if titles is None:
            titles = [f'Gradient {i+1}' for i in range(n_components)]
        
        fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 4))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            im = axes[i].scatter(range(len(gradients)), gradients[:, i], 
                               c=gradients[:, i], cmap=cmap, s=20)
            axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
            axes[i].set_xlabel('ROI Index')
            axes[i].set_ylabel('Gradient Value')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_gradient_comparison(self, gradients1, gradients2, 
                               labels=None, component=0, 
                               title="Gradient Comparison",
                               save_path=None):
        """
        Plot comparison between two sets of gradients.
        
        Parameters
        ----------
        gradients1, gradients2 : np.ndarray
            Gradient arrays to compare
        labels : list, optional
            ROI labels
        component : int, default=0
            Which gradient component to plot
        title : str, default="Gradient Comparison"
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grad1 = gradients1[:, component]
        grad2 = gradients2[:, component]
        
        # Scatter plot
        ax.scatter(grad1, grad2, alpha=0.6, s=30)
        
        # Add diagonal line
        min_val = min(np.min(grad1), np.min(grad2))
        max_val = max(np.max(grad1), np.max(grad2))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(grad1, grad2)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
               transform=ax.transAxes, fontsize=14,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel(f'Gradient {component+1} - Condition 1', fontsize=12)
        ax.set_ylabel(f'Gradient {component+1} - Condition 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_gradient_distribution(self, gradients, component=0, 
                                  title=None, save_path=None):
        """
        Plot distribution of gradient values.
        
        Parameters
        ----------
        gradients : np.ndarray
            Gradient data
        component : int, default=0
            Which component to plot
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        grad_values = gradients[:, component]
        
        # Histogram
        ax.hist(grad_values, bins=30, alpha=0.7, density=True, color='skyblue')
        
        # Add normal distribution overlay
        mu, sigma = np.mean(grad_values), np.std(grad_values)
        x = np.linspace(np.min(grad_values), np.max(grad_values), 100)
        y = ((1 / (sigma * np.sqrt(2 * np.pi))) * 
             np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        ax.plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        ax.set_xlabel(f'Gradient {component+1} Values', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        
        if title is None:
            title = f'Distribution of Gradient {component+1}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax


class StatisticalPlotter:
    """
    Class for plotting statistical results.
    """
    
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
    
    def plot_t_statistics(self, t_stats, p_values=None, alpha=0.05,
                         labels=None, title="T-Statistics",
                         save_path=None):
        """
        Plot t-statistics matrix with significance overlay.
        
        Parameters
        ----------
        t_stats : np.ndarray
            T-statistics matrix
        p_values : np.ndarray, optional
            P-values matrix for significance overlay
        alpha : float, default=0.05
            Significance threshold
        labels : list, optional
            ROI labels
        title : str, default="T-Statistics"
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot t-statistics
        abs_max = np.max(np.abs(t_stats))
        im = ax.imshow(t_stats, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        
        # Add significance overlay if p-values provided
        if p_values is not None:
            significant = p_values < alpha
            y_coords, x_coords = np.where(significant)
            ax.scatter(x_coords, y_coords, s=1, c='black', marker='.')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('T-statistic', fontsize=12)
        
        # Labels
        if labels is not None:
            ax.set_xticks(range(0, len(labels), max(1, len(labels)//10)))
            ax.set_yticks(range(0, len(labels), max(1, len(labels)//10)))
            ax.set_xticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels)//10))], 
                              rotation=90, fontsize=8)
            ax.set_yticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels)//10))], 
                              fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if p_values is not None:
            n_significant = np.sum(p_values < alpha)
            n_total = np.prod(p_values.shape)
            ax.text(0.02, 0.98, f'Significant: {n_significant}/{n_total}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_effect_sizes(self, effect_sizes, labels=None,
                         title="Effect Sizes", save_path=None):
        """
        Plot effect sizes matrix.
        
        Parameters
        ----------
        effect_sizes : np.ndarray
            Effect sizes matrix
        labels : list, optional
            ROI labels
        title : str, default="Effect Sizes"
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot effect sizes
        abs_max = max(np.abs(np.min(effect_sizes)), np.abs(np.max(effect_sizes)))
        im = ax.imshow(effect_sizes, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Cohen's d", fontsize=12)
        
        # Labels
        if labels is not None:
            step = max(1, len(labels)//10)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_yticks(range(0, len(labels), step))
            ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], 
                              rotation=90, fontsize=8)
            ax.set_yticklabels([labels[i] for i in range(0, len(labels), step)], 
                              fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_network_statistics(self, network_stats, network_names, 
                               metric='mean', title=None, save_path=None):
        """
        Plot network-level statistics.
        
        Parameters
        ----------
        network_stats : dict
            Dictionary with network statistics
        network_names : list
            Network names
        metric : str, default='mean'
            Which metric to plot
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data = network_stats[metric]
        
        # Create bar plot
        network_indices = range(len(network_names))
        bars = ax.bar(network_indices, np.diag(data), 
                     color=plt.cm.Set3(np.linspace(0, 1, len(network_names))))
        
        # Add error bars if std is available
        if 'std' in network_stats:
            ax.errorbar(network_indices, np.diag(data), 
                       yerr=np.diag(network_stats['std']),
                       fmt='none', color='black', capsize=5)
        
        ax.set_xlabel('Networks', fontsize=12)
        ax.set_ylabel(f'{metric.capitalize()} Connectivity', fontsize=12)
        ax.set_xticks(network_indices)
        ax.set_xticklabels(network_names, rotation=45, ha='right')
        
        if title is None:
            title = f'Network {metric.capitalize()} Connectivity'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax


def create_summary_figure(connectivity_matrices, gradients, statistical_results,
                         labels=None, save_path=None):
    """
    Create a comprehensive summary figure.
    
    Parameters
    ----------
    connectivity_matrices : dict
        Dictionary with connectivity matrices for different conditions
    gradients : dict
        Dictionary with gradient data for different conditions
    statistical_results : dict
        Dictionary with statistical test results
    labels : list, optional
        ROI labels
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib figure
        Summary figure
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot connectivity matrices
    if 'condition1' in connectivity_matrices and 'condition2' in connectivity_matrices:
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(connectivity_matrices['condition1'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_title('Condition 1 Connectivity')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(connectivity_matrices['condition2'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title('Condition 2 Connectivity')
        plt.colorbar(im2, ax=ax2)
        
        # Plot difference
        ax3 = fig.add_subplot(gs[0, 2])
        diff = connectivity_matrices['condition1'] - connectivity_matrices['condition2']
        abs_max = np.max(np.abs(diff))
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        ax3.set_title('Difference (Cond1 - Cond2)')
        plt.colorbar(im3, ax=ax3)
    
    # Plot gradients
    if 'condition1' in gradients and 'condition2' in gradients:
        ax4 = fig.add_subplot(gs[1, 0])
        grad1 = gradients['condition1'][:, 0]  # First component
        ax4.scatter(range(len(grad1)), grad1, c=grad1, cmap='viridis', s=20)
        ax4.set_title('Condition 1 - Gradient 1')
        ax4.set_xlabel('ROI Index')
        ax4.set_ylabel('Gradient Value')
        
        ax5 = fig.add_subplot(gs[1, 1])
        grad2 = gradients['condition2'][:, 0]
        ax5.scatter(range(len(grad2)), grad2, c=grad2, cmap='viridis', s=20)
        ax5.set_title('Condition 2 - Gradient 1')
        ax5.set_xlabel('ROI Index')
        ax5.set_ylabel('Gradient Value')
        
        # Gradient comparison
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(grad1, grad2, alpha=0.6, s=30)
        min_val = min(np.min(grad1), np.min(grad2))
        max_val = max(np.max(grad1), np.max(grad2))
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        correlation = np.corrcoef(grad1, grad2)[0, 1]
        ax6.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax6.transAxes)
        ax6.set_xlabel('Condition 1 - Gradient 1')
        ax6.set_ylabel('Condition 2 - Gradient 1')
        ax6.set_title('Gradient Correlation')
    
    # Plot statistical results
    if 't_statistics' in statistical_results:
        ax7 = fig.add_subplot(gs[2, 0])
        t_stats = statistical_results['t_statistics']
        abs_max = np.max(np.abs(t_stats))
        im7 = ax7.imshow(t_stats, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        ax7.set_title('T-Statistics')
        plt.colorbar(im7, ax=ax7)
        
        if 'p_values' in statistical_results:
            # Add significance overlay
            p_vals = statistical_results['p_values']
            significant = p_vals < 0.05
            y_coords, x_coords = np.where(significant)
            ax7.scatter(x_coords, y_coords, s=1, c='black', marker='.')
    
    if 'p_values' in statistical_results:
        ax8 = fig.add_subplot(gs[2, 1])
        p_vals = statistical_results['p_values']
        im8 = ax8.imshow(-np.log10(p_vals), cmap='hot')
        ax8.set_title('-log10(p-values)')
        plt.colorbar(im8, ax=ax8)
    
    # Summary statistics
    if 'summary' in statistical_results:
        ax9 = fig.add_subplot(gs[2, 2])
        summary = statistical_results['summary']
        ax9.text(0.1, 0.9, f"Total tests: {summary.get('n_total_tests', 'N/A')}", 
                transform=ax9.transAxes, fontsize=12)
        ax9.text(0.1, 0.8, f"Significant: {summary.get('n_significant', 'N/A')}", 
                transform=ax9.transAxes, fontsize=12)
        ax9.text(0.1, 0.7, f"Proportion: {summary.get('proportion_significant', 'N/A'):.3f}", 
                transform=ax9.transAxes, fontsize=12)
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.set_title('Statistical Summary')
        ax9.axis('off')
    
    plt.suptitle('Psychedelic Gradient Analysis Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully!")
    
    # Create example data for testing
    n_rois = 100
    
    # Example connectivity matrices
    np.random.seed(42)
    conn1 = np.random.rand(n_rois, n_rois)
    conn1 = (conn1 + conn1.T) / 2  # Make symmetric
    np.fill_diagonal(conn1, 1)
    
    conn2 = conn1 + 0.1 * np.random.randn(n_rois, n_rois)
    conn2 = (conn2 + conn2.T) / 2
    np.fill_diagonal(conn2, 1)
    
    # Example gradients
    grad1 = np.random.randn(n_rois, 3)
    grad2 = grad1 + 0.2 * np.random.randn(n_rois, 3)
    
    # Test plotting functions
    plotter = ConnectivityPlotter()
    fig1, ax1 = plotter.plot_connectivity_matrix(conn1, title="Example Connectivity")
    
    grad_plotter = GradientPlotter()
    fig2 = grad_plotter.plot_gradients_on_surface(grad1, n_components=2)
    
    print("Example plots created successfully!")
    plt.show()