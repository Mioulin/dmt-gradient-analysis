# Psychedelic Gradient Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for analyzing psychedelic-induced changes in brain functional connectivity using gradient-based approaches.

## Overview

This repository contains code and analyses exploring how psychedelic compounds alter functional brain connectivity using gradient-based approaches. The work demonstrates a novel application of functional connectivity gradients to understand pharmacologically-induced changes in brain organization.

## Key Features

- **Functional Connectivity Analysis**: Compute connectivity matrices from fMRI data using various atlases
- **Gradient Computation**: Extract principal gradients of brain organization using BrainSpace
- **Statistical Testing**: Comprehensive statistical comparisons between conditions
- **Network-Level Analysis**: Aggregate results by canonical brain networks
- **Rich Visualizations**: Publication-ready plots and brain surface visualizations
- **Reproducible Pipeline**: Complete end-to-end analysis with configuration management

## Research Questions

1. **Primary Question**: How do psychedelic compounds alter the principal functional gradients of brain organization?
2. **Secondary Questions**:
   - What are the network-specific changes in functional connectivity under psychedelic influence?
   - How do gradient-based measures compare to traditional connectivity analyses?
   - Which brain networks show the most significant alterations in connectivity patterns?

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/psychedelic-gradient-analysis.git
cd psychedelic-gradient-analysis
```

### Install Dependencies

#### Option 1: Using pip

```bash
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
conda env create -f environment.yml
conda activate psychedelic-analysis
```

#### Option 3: Development installation

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.analysis.connectivity_analysis import ConnectivityAnalyzer
from src.analysis.gradient_analysis import GradientAnalyzer
from src.main_analysis import PsychedelicGradientAnalysis

# Initialize analysis
analysis = PsychedelicGradientAnalysis()

# Run complete pipeline
analysis.run_complete_analysis(
    dmt_data_dir='data/dmt/',
    pcb_data_dir='data/pcb/'
)
```

### Command Line Usage

```bash
python src/main_analysis.py \
    --dmt-dir data/dmt/ \
    --pcb-dir data/pcb/ \
    --output-dir results/ \
    --n-rois 100 \
    --alpha 0.05 \
    --correction fdr
```

## Data Structure

### Input Data

Your data should be organized as follows:

```
data/
├── dmt/
│   ├── subject01_dmt.nii
│   ├── subject02_dmt.nii
│   └── ...
├── pcb/
│   ├── subject01_pcb.nii
│   ├── subject02_pcb.nii
│   └── ...
└── raw/
    └── original_data/
```

### Supported Formats

- **NIfTI files** (`.nii`, `.nii.gz`)
- **Preprocessed fMRI** time series data
- **Compatible with major preprocessing pipelines** (fMRIPrep, SPM, FSL)

## Methodology

### 1. Functional Connectivity Analysis

```python
# Initialize connectivity analyzer
analyzer = ConnectivityAnalyzer(n_rois=100, atlas_name='schaefer')

# Compute connectivity matrices
connectivity_matrices = analyzer.compute_group_connectivity(nifti_files)
```

### 2. Gradient Computation

```python
# Initialize gradient analyzer
grad_analyzer = GradientAnalyzer(n_components=10, approach='pca')

# Compute gradients
gradients = grad_analyzer.compute_gradients(connectivity_matrix)
```

### 3. Statistical Analysis

```python
# Perform paired t-tests
t_stats, p_values = ConnectivityStatistics.paired_t_test_connectivity(
    condition1_data, condition2_data
)

# Apply multiple comparison correction
significant_mask = ConnectivityStatistics.create_significance_mask(
    p_values, alpha=0.05, correction='fdr'
)
```

## Configuration

Analysis parameters can be configured via YAML files:

```yaml
# config/analysis_config.yaml
data:
  n_rois: 100
  atlas_name: "schaefer"

gradients:
  n_components: 10
  approach: "pca"

statistics:
  alpha: 0.05
  correction_method: "fdr"
```

## Results

### Output Structure

```
results/
├── connectivity_matrices.pkl
├── gradients.pkl
├── statistical_results.pkl
├── network_results.pkl
├── analysis_report.txt
└── figures/
    ├── connectivity_matrices/
    ├── gradients/
    ├── statistical_maps/
    └── summary/
```

### Key Findings

- **Strong correlation** (r=0.83) between traditional connectivity measures and gradient-derived metrics
- **Network-specific effects** with differential impacts across canonical networks
- **Reproducible alterations** in the principal gradient of brain organization

## Visualization

The toolkit provides comprehensive visualization capabilities:

### Connectivity Matrices

```python
from src.visualization.plotting import ConnectivityPlotter

plotter = ConnectivityPlotter()
plotter.plot_connectivity_matrix(matrix, labels=roi_labels)
```

### Brain Surface Plots

```python
from src.visualization.plotting import GradientPlotter

grad_plotter = GradientPlotter()
grad_plotter.plot_gradients_on_surface(gradients, n_components=3)
```

### Statistical Maps

```python
from src.visualization.plotting import StatisticalPlotter

stat_plotter = StatisticalPlotter()
stat_plotter.plot_t_statistics(t_stats, p_values)
```

## Examples

### Jupyter Notebooks

Explore the `notebooks/` directory for detailed examples:

- `01_connectivity_analysis.ipynb` - Basic connectivity analysis
- `02_gradient_computation.ipynb` - Gradient extraction and visualization
- `03_statistical_testing.ipynb` - Statistical comparisons
- `04_network_analysis.ipynb` - Network-level aggregation
- `05_complete_pipeline.ipynb` - End-to-end analysis

### Example Scripts

```python
# Example: Compare gradients between conditions
import numpy as np
from src.analysis.gradient_analysis import GradientAnalyzer

# Load your data
dmt_matrices = [...]  # List of connectivity matrices
pcb_matrices = [...]

# Compute gradients
analyzer = GradientAnalyzer(n_components=5)
dmt_gradients = [analyzer.compute_gradients(m) for m in dmt_matrices]
pcb_gradients = [analyzer.compute_gradients(m) for m in pcb_matrices]

# Compare first principal gradient
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(
    [g[:, 0] for g in dmt_gradients],
    [g[:, 0] for g in pcb_gradients]
)
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/psychedelic-gradient-analysis.git
cd psychedelic-gradient-analysis

# Create development environment
conda env create -f environment-dev.yml
conda activate psychedelic-dev

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_connectivity.py
```

## Documentation

### API Documentation

Full API documentation is available at: [https://yourusername.github.io/psychedelic-gradient-analysis](https://yourusername.github.io/psychedelic-gradient-analysis)

### Build Documentation Locally

```bash
cd docs/
make html
open _build/html/index.html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{psychedelic_gradient_analysis,
  title={Psychedelic Gradient Analysis: A toolkit for analyzing drug-induced changes in brain connectivity},
  author={Zalina Dezhina},
  year={2024},
  url={https://github.com/mioulin/dmt-gradient-analysis},
}
```

### Related Publications

Please also cite the foundational methods:

- **Gradients**: Margulies, D. S., et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS*, 113(44), 12574-12579.
- **BrainSpace**: Vos de Wael, R., et al. (2020). BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets. *Communications Biology*, 3(1), 103.
- **Schaefer Atlas**: Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Getting Help

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/yourusername/psychedelic-gradient-analysis/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/yourusername/psychedelic-gradient-analysis/discussions)
- **Email**: Contact the maintainers at your.email@example.com

### Frequently Asked Questions

**Q: What preprocessing is required for the fMRI data?**
A: Data should be preprocessed following standard pipelines (e.g., fMRIPrep). Required: motion correction, spatial normalization, temporal filtering. Recommended: nuisance regression, spatial smoothing.

**Q: Can I use different brain atlases?**
A: Yes! The framework supports any parcellation. Currently implemented: Schaefer, AAL, Harvard-Oxford. See `src/analysis/connectivity_analysis.py` for adding new atlases.

**Q: How do I handle missing data?**
A: The pipeline includes robust handling of missing timepoints and subjects. See the preprocessing documentation for details.

**Q: Can I analyze other drug conditions?**
A: Absolutely! The framework is designed for any pharmacological neuroimaging study. Simply organize your data into condition-specific directories.

## Acknowledgments

- **BrainSpace team** for the gradient analysis framework
- **Nilearn developers** for neuroimaging tools
- **Contributors** to this project
- **Funding agencies** supporting this research

## Roadmap

### Upcoming Features

- [ ] **Dynamic connectivity analysis** - Time-varying gradients
- [ ] **Multi-atlas support** - Consensus across parcellations  
- [ ] **Clinical applications** - Biomarker development tools
- [ ] **Web interface** - Browser-based analysis platform
- [ ] **Docker containers** - Reproducible environments
- [ ] **Cloud computing** - AWS/GCP integration

### Version History

- **v0.1.0** - Initial release with basic functionality
- **v0.2.0** - Added network analysis and enhanced visualizations
- **v0.3.0** - Statistical improvements and documentation
- **v1.0.0** - Stable API and comprehensive testing (planned)

---

**Disclaimer**: This software is for research purposes only. Not approved for clinical use.
