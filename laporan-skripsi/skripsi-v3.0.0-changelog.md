# CHANGELOG - Proposal Skripsi FSLSM Multi-Label Classification

> **File:** `skripsi-v3.0.0-changelog.md`

Dokumen ini berisi rangkuman perbedaan antara versi proposal skripsi.

---

## [v3.0.0] - 2025-12-16 (skripsi-v3.0.0.md)

### 🔄 Perubahan Major: Ekspansi Algoritma dan Eksperimen

| Aspek | Versi Sebelumnya (v2.0.0) | Versi Terbaru (v3.0.0) |
|-------|---------------------------|------------------------|
| **Jumlah Algoritma** | 3 (XGBoost, RF, SVM) | **5 (+RBF Network, +Self-Training)** |
| **Hyperparameter Tuning** | GridSearchCV | **Nested CV untuk semua 5 algoritma** |
| **Eksperimen Ensemble** | Tidak ada | **Voting Ensemble (Simple & Weighted)** |
| **Best F1-Macro** | 0.7006 | **0.7008** |
| **Notebook** | multi-label-classification-research-review-prod.ipynb | **learning_style_classification_v2.ipynb (31 cells)** |

---

### 📋 Perbandingan Detail

#### BAB I - Pendahuluan

| Komponen | v2.0.0 | v3.0.0 |
|----------|--------|--------|
| Rumusan Masalah | 4 pertanyaan (3 algoritma) | 4 pertanyaan (**5 algoritma + Voting Ensemble**) |
| Batasan Masalah | XGBoost, RF, SVM | **+RBF Network, +Self-Training** |
| Tujuan | 4 tujuan | **5 tujuan (+evaluasi Voting Ensemble)** |

#### BAB II - Tinjauan Pustaka

| Komponen | v2.0.0 | v3.0.0 |
|----------|--------|--------|
| Algoritma | 3 (XGBoost, RF, SVM) | **5 (+RBF Network, +Self-Training)** |
| Teori Baru | - | **Voting Ensemble, RBF Network, Semi-supervised Learning** |
| Nested CV | Disebutkan | **Penjelasan detail (outer/inner loop)** |

#### BAB III - Metode Penelitian

| Komponen | v2.0.0 | v3.0.0 |
|----------|--------|--------|
| Algoritma | 3 | **5 algoritma** |
| Hyperparameter Tuning | GridSearchCV | **Nested CV (10-fold outer, 5-fold inner)** |
| Tuning Scope | RF, XGBoost only | **Semua 5 algoritma dengan param grid masing-masing** |
| Eksperimen Tambahan | - | **Voting Ensemble (Simple & Weighted)** |

#### BAB IV - Hasil dan Pembahasan

| Komponen | v2.0.0 | v3.0.0 |
|----------|--------|--------|
| Algoritma Dievaluasi | 3 | **5** |
| Tabel Perbandingan | 3 baris | **5 baris + ranking emoji** |
| Voting Ensemble | - | **✅ Ada (Simple & Weighted)** |
| Analisis Ensemble | - | **Kesimpulan: Ensemble tidak meningkatkan performa** |
| Hyperparameter Results | - | **Tabel best params untuk semua algoritma** |

#### BAB V - Kesimpulan dan Saran

| Komponen | v2.0.0 | v3.0.0 |
|----------|--------|--------|
| Kesimpulan | 4 poin | **5 poin (+Voting Ensemble finding)** |
| Saran | 4 poin | **4 poin (updated)** |

---

### ✅ Penambahan Baru di v3.0.0

1. **Dua Algoritma Baru**
   - RBF Network (Radial Basis Function Network)
   - Self-Training (Semi-supervised Learning)

2. **Nested Cross-Validation Komprehensif**
   - Diterapkan untuk SEMUA 5 algoritma
   - 10-fold outer, 5-fold inner
   - Parameter grid untuk setiap algoritma

3. **Eksperimen Voting Ensemble**
   - Simple Voting (3 model teratas)
   - Weighted Voting (XGBoost×2, RF×1, Self-Training×1)
   - Kesimpulan: Tidak meningkatkan performa

4. **Notebook Baru (Simplified)**
   - `learning_style_classification_v2.ipynb`
   - 31 cells (vs 72 cells di notebook original)
   - Alur lebih ringkas dan mudah dibaca

5. **Hasil Eksperimen Lengkap**
   - Tabel ranking 5 algoritma dengan F1-Macro, F1-Micro, Subset Accuracy
   - Best params dari Nested CV
   - Perbandingan Voting Ensemble

---

### 📊 Hasil Eksperimen Aktual (v3.0.0)

#### Perbandingan Strategi Imputasi (Tidak Berubah)
| Strategi | F1-Macro | Ranking |
|----------|----------|---------|
| **Median** | **0.6884 ± 0.0834** | **1** |
| Mean | 0.6679 ± 0.0628 | 2 |
| Zero | 0.6413 ± 0.0686 | 3 |
| MICE | 0.6044 ± 0.0721 | 4 |

#### Perbandingan 5 Algoritma (BARU)
| Rank | Algoritma | F1-Macro | F1-Micro | Subset Acc | Status |
|------|-----------|----------|----------|------------|--------|
| 🥇 | **XGBoost** | **0.7008** | 0.7565 | 0.5609 | ✅ BEST |
| 🥈 | Self-Training | 0.6922 | 0.7500 | 0.5478 | ✅ Strong |
| 🥉 | Random Forest | 0.6884 | 0.7457 | 0.5391 | ✅ Strong |
| 4 | RBF Network | 0.4359 | 0.6304 | 0.3130 | ⚠️ Baseline |
| 5 | SVM | 0.4357 | 0.6304 | 0.3130 | ⚠️ Baseline |

#### Voting Ensemble (BARU)
| Method | F1-Macro | vs XGBoost |
|--------|----------|------------|
| XGBoost (Single) | **0.7008** | - |
| Simple Voting | 0.6952 | -0.80% |
| Weighted Voting | 0.6970 | -0.54% |

**Kesimpulan:** Voting Ensemble tidak meningkatkan performa. XGBoost single model tetap terbaik.

---

### ❌ Perubahan dari v2.0.0

1. **Best F1-Macro**
   - v2.0.0: 0.7006
   - v3.0.0: **0.7008** (sedikit lebih tinggi)

2. **Jumlah Algoritma**
   - v2.0.0: 3 algoritma
   - v3.0.0: **5 algoritma**

3. **Hyperparameter Tuning**
   - v2.0.0: Hanya untuk RF dan XGBoost
   - v3.0.0: **Untuk semua 5 algoritma**

---

## Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v3.0.0 | 2025-12-16 | 5 algoritma, Nested CV semua, Voting Ensemble |
| v2.0.0 | 2025-12-14 | Multi-label, 4 imputasi, 3 algoritma |
| v1.0.0 | 2025 | Proposal awal (LSTM) |

---

## Catatan Versioning

- **Major version (X.0.0):** Perubahan metodologi fundamental
- **Minor version (0.X.0):** Penambahan atau modifikasi signifikan
- **Patch version (0.0.X):** Perbaikan minor, typo, formatting

### File Terkait
- `skripsi-v1.0.0.md` - Proposal v1.0.0
- `skripsi-v2.0.0.md` - Proposal v2.0.0
- `skripsi-v3.0.0.md` - Proposal v3.0.0 (CURRENT)
- `skripsi-v3.0.0-changelog.md` - Dokumen ini
