from setuptools import setup, find_packages

setup(
    name="psychedelic-gradient-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Gradient-based analysis of psychedelic-induced changes in brain functional connectivity",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/psychedelic-gradient-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "nibabel>=3.2.0",
        "nilearn>=0.8.0",
        "brainspace>=0.1.2",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "jupyter>=1.0.0"],
        "vis": ["plotly>=5.0.0", "seaborn>=0.11.0"],
    },
)