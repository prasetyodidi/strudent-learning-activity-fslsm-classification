# CHANGELOG - Proposal Skripsi FSLSM Multi-Label Classification

> **File:** `skripsi-v2.0.0-changelog.md`

Dokumen ini berisi rangkuman perbedaan antara versi proposal skripsi.

---

## [v2.0.0] - 2025-12-14 (skripsi-v2.0.0.md)

### 🔄 Perubahan Major: Metodologi Penelitian

| Aspek | Versi Awal (v1.0.0) | Versi Terbaru (v2.0.0) |
|-------|---------------------|------------------------|
| **Pendekatan ML** | Deep Learning (LSTM) | Ensemble Machine Learning (XGBoost, Random Forest) |
| **Tipe Klasifikasi** | Single-label (implisit) | **Multi-label Classification** |
| **Data Input** | Data sekuensial (log klik) | Data agregat waktu interaksi |
| **Framework** | TensorFlow/Keras | scikit-learn, XGBoost |
| **Fokus Penelitian** | Temporal pattern recognition | Imputation strategies & ensemble comparison |

---

### 📋 Perbandingan Detail

#### BAB I - Pendahuluan

| Komponen | v1.0.0 | v2.0.0 |
|----------|--------|--------|
| Latar Belakang | Fokus pada keunggulan LSTM untuk data sekuensial | Fokus pada multi-label classification dan ensemble methods |
| Rumusan Masalah | 3 pertanyaan (mapping FSLSM, model LSTM, perbandingan dengan SVM/ANN) | 4 pertanyaan (+ pengaruh strategi imputasi) |
| Batasan Masalah | LSTM only, 4 dimensi FSLSM | XGBoost/RF/SVM, 2 dimensi FSLSM (Processing, Input) |
| Tujuan | Membangun model LSTM | Membandingkan imputasi & algoritma ensemble |

#### BAB II - Tinjauan Pustaka

| Komponen | v1.0.0 | v2.0.0 |
|----------|--------|--------|
| Landasan Teori | LSTM, RNN, Deep Learning | Multi-Label Classification, Ensemble Methods, Imputation Strategies |
| Referensi Utama | Hochreiter & Schmidhuber (1997), Kalita et al. (2025) | Zhang & Zhou (2014), Chen & Guestrin (2016) |
| Teori Baru | - | Multi-Label Classification, Random Oversampling, Strategi Imputasi |

#### BAB III - Metode Penelitian

| Komponen | v1.0.0 | v2.0.0 |
|----------|--------|--------|
| Dataset | Log interaksi sekuensial | Waktu agregat per jenis materi |
| Preprocessing | Sequence formation untuk LSTM | 4 strategi imputasi (Zero, Mean, Median, MICE) |
| Class Imbalance | Tidak disebutkan | Random Oversampling (123 → 230 samples) |
| Model | LSTM dengan Input/Hidden/Dense/Output layers | XGBoost, Random Forest, SVM dengan MultiOutputClassifier |
| Hyperparameter | epoch, batch size, learning rate, neurons | n_estimators, max_depth, learning_rate, min_samples_leaf |
| Evaluasi | Accuracy, Precision, Recall, F1-Score | **F1-Macro (primary)**, F1-Micro, Hamming Loss, Subset Accuracy |
| Cross-Validation | Train/Test split (80/20) | Stratified K-Fold, Monte Carlo CV, **Nested CV** |

#### BAB IV & V - Hasil dan Kesimpulan

| Komponen | v1.0.0 | v2.0.0 |
|----------|--------|--------|
| Hasil Eksperimen | Belum ada (proposal) | ✅ Ada ringkasan hasil aktual |
| Best Algorithm | - | XGBoost (F1-Macro: 0.7006) |
| Best Imputation | - | Median (F1-Macro: 0.6884) |
| Kesimpulan | - | Berbasis data eksperimen |
| Saran | - | Classifier Chains, penambahan fitur, integrasi ELS |

---

### ✅ Penambahan Baru di v2.0.0

1. **Strategi Imputasi Missing Value**
   - Zero, Mean, Median, MICE
   - Perbandingan performa per strategi

2. **Multi-Label Classification Theory**
   - Binary Relevance
   - Classifier Chains
   - MultiOutputClassifier

3. **Metrik Evaluasi Multi-Label**
   - F1-Macro (metrik utama)
   - Hamming Loss
   - Subset Accuracy

4. **Nested Cross-Validation**
   - Untuk hyperparameter tuning yang unbiased

5. **BAB IV - Hasil Eksperimen (Ringkasan)**
   - Tabel perbandingan strategi imputasi
   - Tabel perbandingan algoritma
   - Validasi dengan benchmark

6. **BAB V - Kesimpulan dan Saran**
   - Kesimpulan berbasis data
   - Saran pengembangan konkret

---

### ❌ Dihapus dari v1.0.0

1. **Deep Learning / LSTM**
   - Tidak sesuai dengan eksperimen yang dilakukan
   - Diganti dengan ensemble methods

2. **Arsitektur Neural Network**
   - Input Layer → Hidden Layer (LSTM) → Dense → Output
   - Diganti dengan tree-based models

3. **TensorFlow/Keras**
   - Diganti dengan scikit-learn dan XGBoost library

4. **Referensi LSTM**
   - Hochreiter & Schmidhuber (1997)
   - Kalita et al. (2025) - Bi-LSTM

---

### 📊 Hasil Eksperimen Aktual (v2.0.0)

#### Perbandingan Strategi Imputasi
| Strategi | F1-Macro | Ranking |
|----------|----------|---------|
| **Median** | **0.6884 ± 0.0834** | **1** |
| Mean | 0.6679 ± 0.0628 | 2 |
| Zero | 0.6413 ± 0.0686 | 3 |
| MICE | 0.6044 ± 0.0721 | 4 |

#### Perbandingan Algoritma
| Algoritma | F1-Macro | Status |
|-----------|----------|--------|
| **XGBoost** | **0.7006** | ✅ BEST |
| Random Forest | 0.6884 | ✅ Strong |
| SVM | 0.4311 | ⚠️ Baseline |

---

## [v1.0.0] - 2025 (skripsi-v1.0.0.md)

### Initial Release
- Proposal awal dengan fokus pada LSTM untuk data sekuensial
- Belum ada hasil eksperimen
- Fokus pada 4 dimensi FSLSM lengkap

---

## Catatan Versioning

- **Major version (X.0.0):** Perubahan metodologi fundamental
- **Minor version (0.X.0):** Penambahan atau modifikasi signifikan
- **Patch version (0.0.X):** Perbaikan minor, typo, formatting

### File Terkait
- `skripsi-v1.0.0.md` - Proposal v1.0.0
- `skripsi-v2.0.0.md` - Proposal v2.0.0
- `skripsi-v2.0.0-changelog.md` - Dokumen ini
- `CHANGELOG.md` - Dokumen ini
