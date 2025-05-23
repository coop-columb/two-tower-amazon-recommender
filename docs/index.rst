Two-Tower Amazon Recommender Documentation
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Overview
--------

This project implements a production-ready two-tower neural network recommendation system 
using the Amazon Reviews 2023 dataset. The system is designed for large-scale product 
recommendation with efficient retrieval capabilities.

Project Status
--------------

**Current Implementation:**

- Data loading infrastructure for Amazon Reviews 2023 dataset
- Advanced preprocessing pipeline with text processing and feature engineering
- Configurable data management system

**Coming Soon:**

- Two-tower model implementation with TensorFlow Recommenders
- Training pipeline with MLflow experiment tracking
- FastAPI serving infrastructure with FAISS/Annoy for efficient retrieval
- Comprehensive evaluation metrics

Architecture
------------

The system follows a modular architecture::

    src/
    ├── data/          # Data loading and preprocessing
    ├── models/        # Two-tower model implementation
    ├── training/      # Training pipeline
    ├── evaluation/    # Metrics and evaluation
    └── serving/       # API and retrieval infrastructure

Getting Started
---------------

1. Install dependencies::

    pip install -e ".[dev]"

2. Download sample data::

    python scripts/data_processing/download_data.py --category All_Beauty --sample-size 10000

3. Explore the data::

    python scripts/data_processing/explore_data.py

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`