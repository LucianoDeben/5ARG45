# Tumor Cell Viability Prediction Using Multimodal Data

## Background and Motivation

Precision medicine is transforming cancer treatment by tailoring therapies to the molecular profiles of individual tumors. This personalized approach seeks to maximize treatment efficacy while minimizing side effects, enabling better outcomes for patients. Transcriptomics data, which provide insights into gene expression changes within tumors, play a critical role in understanding how cells respond to drugs. However, drug characteristics—such as their chemical structure and physicochemical properties—are equally vital for determining treatment success.

This project aims to integrate **perturbation transcriptomics data** and **drug chemical descriptors** to develop a multimodal predictive framework for cancer drug response. By combining these two complementary data types, the project aspires to improve the accuracy of cell viability predictions, supporting the selection of effective therapies for individual patients.

### Vision for the Multimodal Framework

- **Transcriptomics Data**: Capturing the biological response of cancer cells to drug treatments.
- **Chemical Descriptors**: Representing the molecular structure and activity of drugs.
- **Multimodal Integration**: Combining these modalities through advanced deep learning techniques to predict tumor cell viability.

The ultimate goal is to develop an explainable, scalable, and robust predictive tool that can generalize across diverse datasets, providing actionable insights for drug discovery and personalized medicine.

---

## Project Overview

This project is divided into multiple **phases**, each building upon the previous one to gradually increase the predictive power and complexity of the model:

1. **Phase 1 - Baseline with Simple ML Models**  
   Establish benchmarks for predictive performance using standard machine learning regression models on transcriptomics data. This phase will focus on understanding the baseline predictive capability of simpler models and setting the stage for comparisons with more advanced approaches in later phases.

2. **Phase 2 - Exploring Feature Representations**  
   Introduce additional feature engineering methods and explore advanced regression models to improve performance beyond the baseline.

3. **Phase 3 - Multimodal Deep Learning**  
   Integrate transcriptomics data with chemical descriptors into a unified deep learning framework, hypothesizing that this multimodal approach will significantly outperform unimodal models.

4. **Phase 4 - Generalization and Robustness**  
   Test the framework on external datasets to assess its ability to generalize and deliver consistent performance across diverse conditions.

## Notes

- Please note that all hyperparamters and other configuration settings are set in `config.yaml` for syncing between different scripts.
