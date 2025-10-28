"""
Output paths configuration for machine learning project
This file provides consistent paths for all outputs
"""

import os
from pathlib import Path

# Base output directory
OUTPUT_DIR = Path("outputs")

# Data output paths
DATA_DIR = OUTPUT_DIR / "data"
RAW_MERGED_DIR = DATA_DIR / "raw_merged"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Model output paths
MODELS_DIR = OUTPUT_DIR / "models"
RF_CLASSIFIER_DIR = MODELS_DIR / "rf_classifier"
NN_MODEL_DIR = MODELS_DIR / "nn_model"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"

# Report output paths
REPORTS_DIR = OUTPUT_DIR / "reports"
CLASSIFICATION_RESULTS_DIR = REPORTS_DIR / "classification_results"
EVALUATION_METRICS_DIR = REPORTS_DIR / "evaluation_metrics"

# Plot output paths
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure all directories exist
for directory in [
    OUTPUT_DIR, DATA_DIR, RAW_MERGED_DIR, PROCESSED_DIR, FEATURES_DIR,
    MODELS_DIR, RF_CLASSIFIER_DIR, NN_MODEL_DIR, ENSEMBLE_DIR,
    REPORTS_DIR, CLASSIFICATION_RESULTS_DIR, EVALUATION_METRICS_DIR,
    PLOTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Specific output file paths
def get_data_output_path(filename, data_type="processed"):
    """Get path for data output files"""
    if data_type == "raw_merged":
        return RAW_MERGED_DIR / filename
    elif data_type == "processed":
        return PROCESSED_DIR / filename
    elif data_type == "features":
        return FEATURES_DIR / filename
    else:
        return DATA_DIR / filename

def get_model_output_path(filename, model_type="rf_classifier"):
    """Get path for model output files"""
    if model_type == "rf_classifier":
        return RF_CLASSIFIER_DIR / filename
    elif model_type == "nn_model":
        return NN_MODEL_DIR / filename
    elif model_type == "ensemble":
        return ENSEMBLE_DIR / filename
    else:
        return MODELS_DIR / filename

def get_report_output_path(filename, report_type="classification_results"):
    """Get path for report output files"""
    if report_type == "classification_results":
        return CLASSIFICATION_RESULTS_DIR / filename
    elif report_type == "evaluation_metrics":
        return EVALUATION_METRICS_DIR / filename
    else:
        return REPORTS_DIR / filename

def get_plot_output_path(filename):
    """Get path for plot output files"""
    return PLOTS_DIR / filename

# Example usage:
# processed_data_path = get_data_output_path("processed_data.csv", "processed")
# model_path = get_model_output_path("rf_model.joblib", "rf_classifier")
# report_path = get_report_output_path("classification_report.csv", "classification_results")
# plot_path = get_plot_output_path("confusion_matrix.png")