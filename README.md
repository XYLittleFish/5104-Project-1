# Concrete Compressive Strength Analysis

This repository contains the gradient descent implementation and experiment script for project 1.

## Files

- `gradient_descent.py`: gradient descent for univariate and multivariate linear models.
- `experiment.py`: runs the experiments and generates the result tables and figures.

## Setup and execution

The experiments were tested with 64-bit Python 3.14.0.

1. Clone or download this repository and open a terminal in the repository directory.
2. Place `Concrete_Data.xls` alongside the two Python files.
3. Install the dependencies:

   ```sh
   python -m pip install numpy==2.5.3 xlrd==2.0.2 matplotlib==3.11.2 scikit-learn==1.9.1 statsmodels==0.15.0
   ```

4. Run the experiment:

   ```sh
   python experiment.py
   ```

No command-line arguments are required. Each gradient descent model runs for 100,000 updates.

## Outputs

The script creates `results/` beside `experiment.py`, containing five CSV files and five PNG files. Rerunning the experiment overwrites these outputs.

| File | Contents |
|---|---|
| `q3_loss.png` | Training loss for the original and standardized multivariate models |
| `q5_loss.png` | Training loss for all three multivariate configurations, including the log transformation |
| `optimal_metrics.csv`, `optimal_metrics.png` | Training and testing MSE and R² for the three optimal multivariate models |
| `coefficients.csv`, `coefficients.png` | Optimal univariate and multivariate slopes |
| `pvalues.csv`, `pvalues.png` | Feature p-values for the three multivariate configurations |
| `training.csv` | Learning rates, first positive training R² iterations, and final gradient descent metrics |
| `univariate_metrics.csv` | Training and testing metrics for the eight optimal univariate models |
