# Multi-Label Learning Style Classification Project

This project implements and compares various multi-label classification algorithms for predicting learning styles based on material interaction patterns. The implementation follows latest research findings (Dec 2025) and demonstrates comprehensive evaluation of **4 imputation strategies**, oversampling techniques, algorithm performance, and cross-validation strategies.

## 📊 Dataset Overview

### Dataset Origin and Merging Process

**Original Datasets:**
1. **Learning Styles Dataset** (`dfjadi-simplified - dfjadi-simplified.csv`)
   - Total: **604 students** (original FSLSM assessment)
   - Content: Felder-Silverman Learning Style Model (FSLSM) assessment results
   - Dimensions: Processing (Aktif/Reflektif), Input (Visual/Verbal), Persepsi, Pemahaman

2. **Time Tracking Dataset** (`mhs_grouping_by_material_type.csv`)
   - Content: Actual time spent on different learning materials
   - Features: Video time, Document time, Article time, Tasks, Forums, Quizzes

**Merging Process:**
- **Join Type**: Inner join (intersection)
- **Join Key**: Student ID (NIM = NPM)
- **Result**: **123 students** with complete information
- **Match Rate**: 20.4% (123 out of 604 original students)

### Current Dataset (After EDA)
- **Source**: `outputs/data/processed/cleaned_learning_styles_dataset_*.csv`
- **Samples**: **123 students** (complete data)
- **Features**: 6 numerical time-based features
- **Labels**: 4 learning style classes (multi-label classification)
- **Class Imbalance**: Severe imbalance (17.5:1 ratio)
- **Imputation Strategies**: 4 datasets created (Zero, Mean, Median, MICE)

#### Feature Description
| Feature | Description | Type | Data Availability |
|---------|-------------|------|-------------------|
| `time_materials_video` | Time spent on video materials | Numeric (seconds) | ~24 non-zero values |
| `time_materials_document` | Time spent on document materials | Numeric (seconds) | ~61 non-zero values |
| `time_materials_article` | Time spent on article materials | Numeric (seconds) | ~4 non-zero values |
| `time_tasks` | Time spent on tasks | Numeric (seconds) | ~3 non-zero values |
| `time_forums` | Time spent on forums | Numeric (seconds) | ~4 non-zero values |
| `time_quizzes` | Time spent on quizzes | Numeric (seconds) | All zeros |

#### Label Classes (Felder-Silverman Learning Style Model)
- **Aktif**: Active learning style (learning by doing)
- **Reflektif**: Reflective learning style (learning by thinking)
- **Verbal**: Verbal learning style (learning through words)
- **Visual**: Visual learning style (learning through pictures/diagrams)

#### Label Distribution

**Original Dataset (123 samples - Before Oversampling):**
```
Processing Styles:
  Reflektif: 93 samples (75.6%)
  Aktif: 30 samples (24.4%)

Input Styles:
  Verbal: 96 samples (78.0%)
  Visual: 27 samples (22.0%)

Label Combinations:
  [Reflektif, Verbal]    70 samples (56.9%) ← Majority class
  [Aktif, Verbal]        26 samples (21.1%)
  [Reflektif, Visual]    23 samples (18.7%)
  [Aktif, Visual]         4 samples ( 3.3%) ← Minority class

Imbalance Ratio: 17.5:1 (Majority to Minority) ⚠️ SEVERE IMBALANCE
```

**After Random Oversampling (230 samples):**
```
Imbalance Ratio: ~4.5:1 ✅ MUCH IMPROVED
```

## 🔬 Missing Value Handling: Four Imputation Strategies

### Overview
The dataset contains significant missing values (zeros for time-based features = no activity recorded). We implemented **4 imputation strategies** to handle missing values and compare their impact.

### Imputation Strategies Comparison

| Strategy | Description | Approach | Use Case |
|----------|-------------|----------|----------|
| **Zero** | Fill missing with 0 | Missing → 0 | Treats missing as "no engagement" |
| **Mean** | Fill with column mean | 0 → NaN → Mean | Assumes average engagement |
| **Median** | Fill with column median | 0 → NaN → Median | Robust to outliers |
| **MICE** | Multiple Imputation by Chained Equations | Iterative multivariate regression | Captures feature correlations |

### Imputation Statistics Comparison

| Feature | Zero Mean | Mean Imputed | Median Imputed | MICE Mean |
|---------|-----------|--------------|----------------|-----------|
| `time_materials_video` | 913.2s | 4680.1s | 935.3s | 4683.1s |
| `time_materials_document` | 7068.9s | 14253.6s | 10227.8s | 14253.6s |
| `time_materials_article` | 350.2s | 10769.0s | 10798.5s | 6796.8s |
| `time_tasks` | 131.8s | 5405.0s | 7556.2s | 3059.4s |
| `time_forums` | 0.28s | 8.75s | 5.12s | 4.17s |
| `time_quizzes` | 0.0s | NaN | NaN | 0.0s |

### Datasets Created (4 Imputation Strategies)
1. `cleaned_learning_styles_dataset_zero.csv` - Zero imputation
2. `cleaned_learning_styles_dataset_mean.csv` - Mean imputation
3. `cleaned_learning_styles_dataset_median.csv` - Median imputation
4. `cleaned_learning_styles_dataset_mice.csv` - MICE imputation

## 🔄 Comprehensive Oversampling Analysis & Results

### Multi-Technique Oversampling Implementation
Implemented Random Oversampling for multi-label classification across all 4 imputation strategies.

### Oversampling Results by Imputation Strategy

| Imputation | Original | After Oversampling | Increase |
|------------|----------|-------------------|----------|
| **Zero** | 123 | 230 | +87% |
| **Mean** | 123 | 230 | +87% |
| **Median** | 123 | 230 | +87% |
| **MICE** | 123 | 230 | +87% |

### Balanced Datasets Created
1. `best_balanced_dataset_zero.csv` - Zero imputation + Oversampling
2. `best_balanced_dataset_mean.csv` - Mean imputation + Oversampling
3. `best_balanced_dataset_median.csv` - Median imputation + Oversampling
4. `best_balanced_dataset_mice.csv` - MICE imputation + Oversampling

## 🤖 Experiment Results (Updated Dec 2025)

### 🏆 Best Model: XGBoost
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Performance**: **0.7006 F1-Macro**
- **Validation**: Outperformed Random Forest and SVM in comprehensive benchmarks.

### 📊 Best Imputation Strategy: Median
Based on Random Forest comparisons across all strategies:
1.  **Median**: **0.6884 ± 0.0834** (Best Performer)
2.  Mean: 0.6679 ± 0.0628
3.  Zero: 0.6413 ± 0.0686
4.  MICE: 0.6044 ± 0.0721

**Insight**: Median imputation proved most effective, likely by preserving the central tendency of student engagement times while being robust to outliers (extreme study times).

### Algorithm Performance Summary
| Algorithm | Best F1-Macro | Status |
|-----------|---------------|--------|
| **XGBoost** | **0.7006** | ✅ **BEST** |
| Random Forest | 0.6884 | ✅ Strong |
| SVM | 0.4311 | ⚠️ Baseline |

*Note: Results based on Stratified K-Fold Cross-Validation on the optimal dataset.*

## 📁 Project Structure

```
├── dataset/                     # Raw data files
├── outputs/                     # Generated artifacts
│   ├── data/                    # Processed datasets
│   │   ├── processed/           # Cleaned/Imputed data
│   │   └── balanced/            # Oversampled balanced data
│   ├── models/                  # Saved models (.pkl) & metadata
│   ├── plots/                   # Visualization figures
│   └── reports/                 # JSON summaries & text reports
├── Skripsi_Playground.ipynb     # Initial exploration
├── EDA_Analysis.ipynb           # Data cleaning & Imputation
├── multi-label-oversampling-techniques-comparison.ipynb # Oversampling
├── multi-label-classification-research-review-prod.ipynb # Main Training & Evaluation
├── README.md                    # This documentation
└── requirements.txt             # Dependencies
```

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline
Run the notebooks in the following order:
1. `EDA_Analysis.ipynb`: Preprocess data and generate imputed datasets.
2. `multi-label-oversampling-techniques-comparison.ipynb`: Create balanced datasets.
3. `multi-label-classification-research-review-prod.ipynb`: Train models and generate results.

### 3. Dependencies
- pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- matplotlib, seaborn (for visualization)
- iterative-imputer (sklearn.impute)

## 📚 References
1. Felder, R. M., & Silverman, L. K. (1988). Learning and teaching styles in engineering education.
2. Chen et al. (2023). XGBoost for Multi-label Classification.
3. Zhang & Zhou (2014). A Review on Multi-Label Learning Algorithms.