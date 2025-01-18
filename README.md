# MNIST Classification and Regression Project

This repository contains a machine learning project focused on handwritten digit classification using the MNIST dataset. The project involves implementing and comparing multiple classification methods, including:

- **Binary Logistic Regression** (One-vs-All strategy)
- **Multi-class Logistic Regression** (Softmax function)
- **Support Vector Machines (SVMs)** with various kernel configurations

## Key Features
- Preprocessing of MNIST data: Feature selection, normalization, and partitioning into training, validation, and test sets.
- Implementation of **Binary Logistic Regression** with gradient-based optimization for multi-class classification.
- Optional implementation of **Multi-class Logistic Regression** for a streamlined approach to multi-class classification (extra credit).
- Utilization of `scikit-learn`'s SVM module with analysis of different kernel configurations and hyperparameters.
- Visualizations and performance metrics for comparative evaluation.

## Usage
- Run `script.py` to train and evaluate the models.
- The preprocessing function handles data preparation, ensuring consistent input for all methods.
- Analyze experimental results and model accuracy using the included report.

## Dependencies
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

## How to Run
1. Ensure the `mnist_all.mat` file is available in the repository's root directory.
2. Install the required Python dependencies.
3. Execute `script.py` to preprocess data, train models, and display results.

## Highlights
- In-depth comparison of classification techniques with MNIST.
- Hands-on implementation of logistic regression algorithms.
- Analysis of SVM performance under various kernel and parameter settings.
