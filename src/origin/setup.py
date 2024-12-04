from setuptools import find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "A Python package for multimodal and multi-omics deep learning"
LONG_DESCRIPTION = (
    "Origin is a Python package designed for unified molecular, "
    "genomic, and proteomic data representation. It facilitates "
    "multimodal and multi-omics deep learning workflows with robust "
    "data handling and feature engineering capabilities."
)

# Setting up
setup(
    name="origin",  # The package name
    version=VERSION,
    author="Your Name",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/origin",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",  # PyTorch for deep learning
        "torch-geometric>=2.3.0",  # PyTorch Geometric for GNNs
        "rdkit-pypi",  # RDKit for molecular representation
        "deepchem>=2.7.1",  # DeepChem for cheminformatics
        "numpy>=1.21.0",  # For numerical computations
        "pandas>=1.3.0",  # For DataFrame manipulations
        "scikit-learn>=1.0",  # For preprocessing and metrics
        "matplotlib>=3.4",  # For plotting and visualizations
        "seaborn>=0.11",  # For statistical data visualization
    ],
    keywords=[
        "python",
        "deep learning",
        "bioinformatics",
        "multiomics",
        "cheminformatics",
        "deepchem",
        "torch-geometric",
        "rdkit",
        "protein modeling",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    # project_urls={
    #     "Documentation": "https://github.com/your-repo/origin/docs",
    #     "Source": "https://github.com/your-repo/origin",
    #     "Tracker": "https://github.com/your-repo/origin/issues",
    # },
)
