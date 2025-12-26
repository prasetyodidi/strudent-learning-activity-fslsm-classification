# CHANGELOG - Proposal Skripsi FSLSM Multi-Label Classification

> **File:** `skripsi-v4.0.0-changelog.md`  
> **Version:** 4.0.0  
> **Last Updated:** December 26, 2024

Dokumen ini berisi rangkuman perbedaan antara versi proposal skripsi.

---

## Update History

### Version 4.0.0c (December 26, 2024) - Random Forest Best Model Update

**Critical Update - Best Model Changed:**
- ❌ Previous: XGBoost (0.7008) as best model
- ✅ Updated: **Random Forest (0.7063)** as best model after Nested CV tuning

**Changes Made:**

1. **Section D - Konsep Penelitian:**
   - Changed main algorithm from XGBoost to Random Forest
   - Updated hyperparameter table to Random Forest parameters
   - Added explanation of +2.6% improvement from tuning

2. **Section BAB IV - Hasil dan Pembahasan:**
   - Section D.1 already showed correct default results (XGBoost 0.7008 best)
   - Section E.1 already correctly shows Random Forest (0.7063 best after tuning)
   - Added "Default Parameters" context to distinguish sections

3. **BAB V - Kesimpulan:**
   - Updated imputation F1-Macro value (0.6884 → 0.7083)
   - Random Forest correctly identified as final best model

4. **BAB I & BAB II - Scope Refined:**
   - Removed all references to **Voting Ensemble** from BAB I (Pendahuluan) and BAB II (Tinjauan Pustaka).
   - Adjusted Rumusan Masalah, Tujuan Penelitian, and Manfaat Penelitian to focus on the 5 core algorithms.
   - Renumbered sections in BAB II following the removal of the Voting Ensemble section.
   - Fixed numbering and removed duplicate items in Batasan Masalah.

**Rationale:**
Nested Cross-Validation hyperparameter tuning changed the ranking:
- Before tuning: XGBoost > Self-Training > Random Forest
- After tuning: **Random Forest > XGBoost > Self-Training**

Random Forest showed the largest improvement (+2.6%) from tuning.

---

### Version 4.0.0b (December 23, 2024) - Chapter 4 Comprehensive Refinement

**Based on User Feedback:**
- ✅ Removed flowchart from Chapter 4
- ✅ Added detailed dataset integration explanation from `Skripsi_Playground.ipynb`
- ✅ Expanded preprocessing experiments referencing `EDA_Analysis.ipynb`
- ✅ Detailed oversampling methods from `multi-label-oversampling-techniques-comparison.ipynb`
- ✅ Removed Voting Ensemble section (did not show improvement)
- ✅ Removed "F. Pembahasan" section
- ✅ Expanded "E. Evaluasi Model" to show ALL algorithm results with detailed analysis

**Section A - Pengumpulan Data:**
- Added complete structure of FSLSM dataset (12 columns with table)
- Added complete structure of ELS interaction dataset (8 columns with table)
- Detailed 3-step integration process (standardization → inner join → feature selection)
- Explained match rate reason (20.4% = 123/604)
- Clarified which features are used (3) vs excluded (3)

**Section C - Penanganan Class Imbalance:**
- Added "Eksperimen Teknik Oversampling" subsection
- Explained why Random Oversampling was chosen (4 reasons)
- Listed alternatives considered: MLSMOTE, ADASYN
- Detailed configuration: 1.5x sampling ratio, random duplication with replacement
- Enhanced results table with specific numbers (107 sampel added, 74% improvement)

**Section E - Evaluasi Model:**
- **MAJOR EXPANSION:** From 1 model (XGBoost) to comprehensive analysis of ALL 5 algorithms
- Added ranking table with medals (🥇🥈🥉)
- Created per-algorithm analysis subsections (a-e):
  - XGBoost: Detailed why it won (4 keunggulan)
  - Self-Training: Runner-up analysis (gap -1.2%)
  - Random Forest: Rank 3 with improvement tracking
  - RBF Network: Explained limitations (gap -37.8%)
  - SVM: Baseline role explanation (gap -37.9%)
- Removed Voting Ensemble comparison (moved to removed section)
- Enhanced benchmark validation (3 references)
- Added "Kesimpulan Evaluasi" with gap performance summary

**Section F - Pembahasan:**
- **REMOVED ENTIRELY** as per user request

---

### Version 4.0.0a (December 23, 2024) - Initial v4.0.0 Release

#### 🔄 Perubahan Filosofis: Struktur Bab 3 vs Bab 4

Versi ini merestrukturisasi laporan untuk membedakan antara "Solusi yang Diusulkan" (Bab 3) dan "Eksplorasi Eksperimental" (Bab 4), sesuai arahan untuk menyajikan cerita yang lebih kohesif.

| Aspek | Bab 3 (Metode Penelitian) | Bab 4 (Hasil dan Pembahasan) |
|-------|---------------------------|------------------------------|
| **Fokus** | **Solusi Optimal (The "Best Path")** | **Eksplorasi Lengkap (All Experiments)** |
| **Algoritma** | Hanya menyebut **XGBoost** (pemenang) | Membandingkan **5 Algoritma** (XGBoost, RF, SVM, RBF, Self-Training) |
| **Imputasi** | Fokus pada **Median** (strategi terbaik) | Membandingkan **4 Strategi** (Median, Mean, Zero, MICE) |
| **Hyperparameter** | Menampilkan **Best Values** (konkret) | Menampilkan proses tuning dan parameter grid |
| **Narasi** | "Kami menggunakan X karena..." | "Kami mencoba A, B, C, dan X adalah yang terbaik karena..." |

---

#### 📋 Perbandingan Detail

**BAB III - Metode Penelitian**

| Komponen | v3.0.0 (Sebelumnya) | v4.0.0 (Sekarang) |
|----------|---------------------|-------------------|
| **Algoritma** | Daftar 5 algoritma kandidat | **XGBoost (Extreme Gradient Boosting)** |
| **Alasan** | Eksperimen komparatif | Superioritas performa dan efisiensi |
| **Tuning** | Parameter Grid (Search Space) | **Optimized Hyperparameters (Fix Values)** |
| **Detail Params** | Range (e.g., max_depth=[3,6,9]) | **Fixed (e.g., max_depth=6)** |

**BAB IV - Hasil dan Pembahasan (Tetap Komprehensif)**

Tidak ada pengurangan konten di Bab 4. Bab ini tetap berfungsi sebagai bukti empiris yang memvalidasi keputusan di Bab 3.

- **Imputasi:** Tabel perbandingan 4 metode (Median menang dengan 0.7083 F1).
- **Algoritma:** Tabel ranking 5 model (XGBoost menang dengan 0.7008 F1).
- **Ensemble:** Bukti bahwa Voting Ensemble tidak lebih baik dari Single XGBoost, menjustifikasi penggunaan single model di Bab 3.

---

#### ✅ Data Spesifik yang Diperbarui

1. **Hyperparameter Optimal (Bab 3)**
   - `n_estimators`: 150
   - `max_depth`: 6
   - `learning_rate`: 0.1
   - `gamma`: 0
   - `subsample`: 0.8

2. **Skor Imputasi (Bab 4)**
   - Diperbarui agar presisi sesuai notebook `learning_style_classification_v2.ipynb`:
   - Median: **0.7083** (sebelumnya tercatat 0.6884 dari RF baseline, sekarang disesuaikan)

---

## Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| **v4.0.0b** | 2024-12-23 | Chapter 4 comprehensive refinement based on user feedback |
| **v4.0.0a** | 2024-12-23 | Restrukturisasi Bab 3 (Optimal) vs Bab 4 (Eksplorasi) |
| v3.0.0 | 2024-12-16 | 5 algoritma, Nested CV semua, Voting Ensemble |
| v2.0.0 | 2024-12-14 | Multi-label, 4 imputasi, 3 algoritma |
| v1.0.0 | 2024 | Proposal awal (LSTM) |
