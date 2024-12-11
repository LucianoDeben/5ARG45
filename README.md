# Multimodal Deep Learning for Predicting Cancer Drug Response

This repository contains the implementation of a multimodal deep learning framework for predicting cancer drug response by integrating perturbation transcriptomics data and drug chemical descriptors. The project follows a systematic progression from simple ML benchmarks to advanced multimodal DL models.

---

## **Project Overview**

- **Objective:** Predict cancer cell viability in response to drug treatments using transcriptomics and drug descriptors.
- **Milestones:** Establish baseline results, incrementally increase model complexity, and develop a multimodal framework.
- **Status:** In progress.

---

## **Step-by-Step Plan**

### **Phase 1: Reproducing Baseline Results with Simple ML Models**

- [x] **Task 1.1:** Set up the project repository and environment.
  - Install required packages (e.g., `scikit-learn`, `pandas`, `numpy`).
  - Load and preprocess the gene expression derived TF dataset.
- [x] **Task 1.2:** Implement Ridge and Lasso regression models.
  - Train models on the TF dataset.
  - Evaluate models using metrics (e.g., RMSE, R²).
- [ ] **Task 1.3:** Compare results with earlier research.
  - Report findings in a baseline results summary.

---

### **Phase 2: Developing Unimodal Deep Learning Models**

- [ ] **Task 2.1:** Design a pipeline for TF or gene expression data.
  - Preprocess high-dimensional data for DL models.
  - Apply dimensionality reduction (e.g., PCA, autoencoders) if necessary.
- [ ] **Task 2.2:** Implement unimodal DL models.
  - Train neural networks (e.g., MLPs, CNNs) on the TF dataset.
  - Evaluate model performance using the same metrics as in Phase 1.
- [ ] **Task 2.3:** Compare unimodal DL results with baseline.
  - Highlight improvements and challenges in a summary report.

---

### **Phase 3: Multimodal Modeling with Drug Descriptors**

- [ ] **Task 3.1:** Preprocess and integrate drug descriptor data.
  - Represent drugs using molecular descriptors or SMILES strings.
  - Prepare data loaders for both transcriptomics and drug datasets.
- [ ] **Task 3.2:** Develop the multimodal integration framework.
  - Combine transcriptomics and drug data using vector concatenation.
  - Experiment with attention mechanisms or transformers.
- [ ] **Task 3.3:** Train and evaluate the multimodal model.
  - Assess model performance on internal and external datasets.
  - Compare multimodal results with unimodal benchmarks.

---

### **Phase 4: Validation and Results Analysis**

- [ ] **Task 4.1:** Test on external datasets (e.g., MIX-Seq, clinical data).
- [ ] **Task 4.2:** Perform interpretability analysis.
  - Identify critical genes and chemical descriptors influencing predictions.
- [ ] **Task 4.3:** Finalize visualizations for results.
  - Generate plots for comparison and interpretability insights.

---

## **Expected Deliverables**

1. **Codebase:** Fully implemented models for all phases.
2. **Baseline Results Report:** Analysis of Ridge and Lasso regression performance.
3. **Model Comparisons:** Evaluation of unimodal and multimodal frameworks.
4. **Final Report:** Comprehensive documentation of methods, results, and insights.
5. **Presentation:** Slides summarizing key findings and future directions.

---

## **How to Use This Repository**

1. Clone the repository:

   ```bash
   git clone https://github.com/LucianoDeben/5ARG45.git
   ```

### Key Features of This README

1. **Step-by-Step Tasks**: Tasks are divided into phases, with a logical progression from simple to complex models.
2. **Progress Tracking**: Checkboxes allow for real-time tracking of milestones.
3. **Deliverables and Usage**: Provides clarity on what to expect and how to use the repository.
4. **Future Extendability**: Each phase is modular, making it easier to adapt as the project evolves.

Let me know if you'd like further adjustments!
