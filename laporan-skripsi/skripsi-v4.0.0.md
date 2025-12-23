---
title: "Proposal Skripsi - Prediksi Gaya Belajar FSLSM dengan Multi-Label Classification"
version: "3.0.0"
date: "2025-12-16"
author: "Didi Prasetyo"
status: "Draft"
previous_version: "skripsi-v2.0.0.md"
changelog: "skripsi-v3.0.0-changelog.md"
---

# BAB I PENDAHULUAN

## A. Latar Belakang Masalah

Perkembangan Learning Analytics dan Educational Data Mining (EDM) telah memberikan kontribusi besar dalam dunia pendidikan modern. Dengan memanfaatkan data interaksi mahasiswa dalam sistem pembelajaran elektronik, pendidik dapat memahami pola belajar, memprediksi hasil akademik, serta merancang strategi pembelajaran yang lebih personal (Mian et al., 2022). Penerapan analitik ini menjadi semakin relevan di era digital, di mana proses belajar tidak lagi terbatas pada ruang kelas fisik, tetapi juga berlangsung melalui platform Electronic Learning System (ELS).

Salah satu aspek penting dalam personalisasi pembelajaran adalah memahami gaya belajar mahasiswa. Model Felder-Silverman Learning Style Model (FSLSM) banyak digunakan untuk mengklasifikasikan gaya belajar berdasarkan beberapa dimensi, seperti aktif atau reflektif, sensitif atau intuitif, visual atau verbal, serta sequential atau global (Hasibuan, 2025). Dengan memahami preferensi belajar tersebut, pendidik dapat menyesuaikan materi dan metode penyampaian sehingga meningkatkan keterlibatan dan hasil belajar mahasiswa.

Penelitian sebelumnya telah mengkombinasikan FSLSM dengan berbagai algoritma machine learning klasik seperti Logistic Regression, K-Nearest Neighbor, Random Forest, dan Support Vector Machine (SVM) (Baihaqi, 2024). Hasilnya menunjukkan peningkatan akurasi dalam prediksi gaya belajar, terutama ketika disertai teknik feature selection dan penanganan data imbalance. Namun, penelitian tersebut umumnya menggunakan pendekatan single-label classification, padahal dalam kenyataannya seorang mahasiswa dapat memiliki kombinasi beberapa gaya belajar sekaligus (multi-label).

Dalam konteks Universitas Amikom Purwokerto, data log aktivitas mahasiswa pada Electronic Learning System (ELS) menjadi sumber yang sangat potensial untuk dianalisis. Pola waktu akses materi video, dokumen, artikel, forum diskusi, hingga penyelesaian tugas dapat memberikan gambaran yang kaya mengenai gaya belajar mahasiswa. Oleh karena itu, diperlukan pendekatan **multi-label classification** yang mampu memprediksi kombinasi gaya belajar secara simultan.

Penelitian ini membandingkan **lima algoritma** klasifikasi: XGBoost, Random Forest, SVM, RBF Network, dan Self-Training. Selain itu, juga dilakukan eksperimen **Voting Ensemble** untuk menggabungkan prediksi dari model-model terbaik. Dengan mengkombinasikan teknik penanganan missing value (imputasi), oversampling untuk mengatasi class imbalance, serta **Nested Cross-Validation** untuk hyperparameter tuning yang unbiased, penelitian ini berupaya membangun model prediksi gaya belajar mahasiswa yang optimal.

## B. Rumusan Masalah

1. Bagaimana memetakan aktivitas mahasiswa dalam ELS ke dimensi FSLSM menggunakan pendekatan multi-label classification?
2. Bagaimana pengaruh strategi imputasi missing value (Zero, Mean, Median, MICE) terhadap performa model prediksi gaya belajar?
3. Bagaimana performa lima algoritma (XGBoost, Random Forest, SVM, RBF Network, Self-Training) dalam tugas multi-label classification?
4. Apakah Voting Ensemble dapat meningkatkan performa dibandingkan model tunggal terbaik?

## C. Batasan Masalah

1. Dataset terfokus pada ELS Amikom Purwokerto dengan data waktu interaksi mahasiswa terhadap berbagai jenis materi pembelajaran.
2. Model machine learning yang digunakan: XGBoost, Random Forest, SVM, RBF Network, dan Self-Training.
3. Gaya belajar mengacu pada dua dimensi utama FSLSM: Processing (Aktif/Reflektif) dan Input (Visual/Verbal).
4. Pendekatan klasifikasi menggunakan multi-label classification dengan MultiOutputClassifier.
5. Hyperparameter tuning menggunakan Nested Cross-Validation (10-fold outer, 5-fold inner).

## D. Tujuan Penelitian

1. Menerapkan FSLSM dalam analisis gaya belajar berbasis data ELS dengan pendekatan multi-label.
2. Membandingkan efektivitas empat strategi imputasi missing value terhadap performa model.
3. Mengembangkan dan membandingkan lima algoritma prediksi gaya belajar.
4. Mengevaluasi efektivitas Voting Ensemble dibandingkan model tunggal.
5. Mengevaluasi model dengan metrik standar multi-label classification (F1-Macro, Hamming Loss, Subset Accuracy).

## E. Manfaat Penelitian

1. **Teoritis:** Kontribusi pada penelitian Learning Analytics berbasis multi-label classification dengan perbandingan komprehensif lima algoritma dan teknik ensemble.
2. **Praktis:** Mendukung personalisasi pembelajaran di Amikom Purwokerto dengan menyediakan model prediksi gaya belajar yang dapat diintegrasikan ke dalam ELS.

---

# BAB II TINJAUAN PUSTAKA

## A. Landasan Teori

### 1. Learning Analytics dan Educational Data Mining (EDM)

Learning Analytics adalah proses pengumpulan, analisis, dan pelaporan data mengenai mahasiswa dan konteks pembelajaran untuk memahami serta mengoptimalkan proses belajar. Sementara itu, Educational Data Mining (EDM) fokus pada penerapan teknik data mining untuk mengeksplorasi pola tersembunyi dalam data pendidikan.

### 2. Electronic Learning System (ELS)

ELS merupakan sistem yang digunakan untuk mendukung pembelajaran daring di Universitas Amikom Purwokerto. Sistem ini mencatat berbagai aktivitas mahasiswa, termasuk waktu yang dihabiskan untuk mengakses materi video, dokumen, artikel, forum diskusi, dan kuis.

### 3. Gaya Belajar dan Felder-Silverman Learning Style Model (FSLSM)

Model FSLSM yang dikembangkan oleh Felder dan Silverman (1988) mengklasifikasikan gaya belajar mahasiswa ke dalam empat dimensi:

- **Processing:** Aktif → Reflektif
- **Perception:** Sensitif → Intuitif
- **Input:** Visual → Verbal
- **Understanding:** Sequential → Global

Penelitian ini berfokus pada dua dimensi: **Processing** (Aktif/Reflektif) dan **Input** (Visual/Verbal).

### 4. Multi-Label Classification

Multi-label classification memungkinkan satu instance memiliki beberapa label sekaligus. Metode umum meliputi:
- **Binary Relevance:** Classifier independen untuk setiap label
- **Classifier Chains:** Memodelkan dependensi antar label
- **MultiOutputClassifier:** Wrapper scikit-learn untuk Binary Relevance

### 5. Algoritma yang Digunakan

| Algoritma | Deskripsi | Keunggulan |
|-----------|-----------|------------|
| **XGBoost** | Extreme Gradient Boosting | Handling missing value, regularization |
| **Random Forest** | Ensemble decision trees | Robust, feature importance |
| **SVM** | Support Vector Machine | Effective for small datasets |
| **RBF Network** | Radial Basis Function Network | Non-linear pattern recognition |
| **Self-Training** | Semi-supervised learning | Leverage unlabeled data patterns |

### 6. Voting Ensemble

Voting Ensemble menggabungkan prediksi dari beberapa model untuk menghasilkan prediksi akhir. Dua metode yang digunakan:
- **Simple Voting:** Mayoritas suara dari semua model
- **Weighted Voting:** Memberikan bobot lebih tinggi pada model terbaik

### 7. Penanganan Missing Value: Strategi Imputasi

| Strategi | Deskripsi | Karakteristik |
|----------|-----------|---------------|
| **Zero** | Missing → 0 | Menganggap tidak ada aktivitas |
| **Mean** | Missing → Mean kolom | Asumsi aktivitas rata-rata |
| **Median** | Missing → Median kolom | Robust terhadap outlier |
| **MICE** | Iterative imputation | Menangkap korelasi antar fitur |

### 8. Nested Cross-Validation

Nested CV mencegah bias dalam estimasi performa dengan memisahkan proses hyperparameter tuning dari evaluasi akhir:
- **Outer loop (10-fold):** Estimasi performa model
- **Inner loop (5-fold):** Pemilihan hyperparameter terbaik

## B. Penelitian Sebelumnya

1. **Baihaqi dkk. (2024)** - Framework prediksi gaya belajar berbasis FSLSM dengan ANN, akurasi 97%, pendekatan single-label.

2. **Sayed dkk. (2024)** - Prediksi gaya belajar dengan dataset OULAD menggunakan LR, KNN, RF, SVM.

3. **Chen et al. (2023)** - XGBoost efektif untuk dataset kecil-menengah dengan performa konsisten.

4. **Zhang & Zhou (2014)** - Review multi-label learning algorithms sebagai fondasi teoretis.

---

# BAB III METODE PENELITIAN

## A. Tempat dan Waktu Penelitian

Penelitian ini dilakukan di Universitas Amikom Purwokerto dengan data dari Electronic Learning System (ELS) periode akademik 2024-2025.

## B. Metode Pengumpulan Data

1. **Ekstraksi Data Log** - Waktu interaksi mahasiswa pada materi video, dokumen, artikel
2. **Dokumentasi FSLSM** - Hasil assessment sebagai ground truth
3. **Integrasi Dataset** - Inner join berdasarkan ID mahasiswa

## C. Alat dan Bahan Penelitian

### 1. Perangkat Keras
- Laptop: Lenovo Ideapad Slim 3, AMD Ryzen 3, 12GB RAM, 512GB SSD

### 2. Perangkat Lunak
- Python 3.10+, scikit-learn, XGBoost, imbalanced-learn

### 3. Dataset
| Aspek | Nilai |
|-------|-------|
| Sampel Awal | 604 mahasiswa |
| Setelah Inner Join | 123 mahasiswa |
| Setelah Oversampling | 230 mahasiswa |
| Fitur Digunakan | 3 (video, document, article time) |
| Label | 4 (Aktif, Reflektif, Visual, Verbal) |

## D. Konsep Penelitian

### 1. Preprocessing Data

#### a. Strategi Imputasi Missing Value
- Median

#### b. Label Encoding
- MultiLabelBinarizer untuk format binary matrix

#### c. Normalisasi
- StandardScaler untuk fitur numerik

### 2. Penanganan Class Imbalance

- **Imbalance Ratio Awal:** 17.5:1
- **Teknik:** Random Oversampling (123 → 230 samples)
- **Imbalance Ratio Akhir:** ~4.5:1

### 3. Pemodelan Machine Learning

#### a. Lima Algoritma
Menggunakan algoritma XGBoost dengan MultiOutputClassifier

#### b. Hyperparameter Tuning
Algiritma XGBoost menggunakan hyperparameter tuning dengan detail parameter sebagai berikut:
| Parameter | Rentang Nilai (Grid Search) |
|-----------|-----------------------------|
| `n_estimators` | 150 |
| `max_depth` | 3, 6, 9 |
| `learning_rate` | 0.05, 0.1 |
| `gamma` | 0, 0.1 |
| `subsample` | 0.8 |

### 4. Evaluasi Model

#### a. Cross-Validation
Evaluasi dilakukan menggunakan teknik **Nested Cross-Validation** dengan skema:
- **Outer Loop (10-Fold):** Membagi data menjadi 10 bagian untuk evaluasi performa model yang tidak bias (unbiased evaluation).
- **Inner Loop (5-Fold):** Dilakukan pada setiap fold outer untuk mencari kombinasi hyperparameter terbaik (grid search).
Pendekatan ini memastikan bahwa data yang digunakan untuk testing benar-benar terpisah dari proses tuning parameter.

#### b. Metrik Evaluasi
| Metrik | Deskripsi |
|--------|-----------|
| **F1-Macro** | Rata-rata F1 per label (metrik utama) |
| **F1-Micro** | F1 global berdasarkan total TP/FP/FN |
| **Hamming Loss** | Fraksi label salah diprediksi |
| **Subset Accuracy** | Prediksi tepat semua label |

---

# BAB IV HASIL DAN PEMBAHASAN

## A. Hasil Eksperimen

### 1. Perbandingan Strategi Imputasi

| Strategi | F1-Macro | Std | Ranking |
|----------|----------|-----|---------|
| **Median** | **0.6884** | ±0.0834 | **1** |
| Mean | 0.6679 | ±0.0628 | 2 |
| Zero | 0.6413 | ±0.0686 | 3 |
| MICE | 0.6044 | ±0.0721 | 4 |

**Temuan:** Median Imputation terbaik karena robust terhadap outlier.

### 2. Perbandingan Lima Algoritma

| Rank | Algoritma | F1-Macro | F1-Micro | Subset Accuracy |
|------|-----------|----------|----------|-----------------|
| 🥇 | **XGBoost** | **0.7008** | 0.7565 | 0.5609 |
| 🥈 | Self-Training | 0.6922 | 0.7500 | 0.5478 |
| 🥉 | Random Forest | 0.6884 | 0.7457 | 0.5391 |
| 4 | RBF Network | 0.4359 | 0.6304 | 0.3130 |
| 5 | SVM | 0.4357 | 0.6304 | 0.3130 |

**Temuan:** XGBoost menunjukkan performa terbaik dengan F1-Macro 0.7008.

### 3. Eksperimen Voting Ensemble

| Method | F1-Macro | vs XGBoost |
|--------|----------|------------|
| XGBoost (Single) | **0.7008** | - |
| Simple Voting | 0.6952 | -0.80% |
| Weighted Voting | 0.6970 | -0.54% |

**Temuan:** Voting Ensemble TIDAK meningkatkan performa. XGBoost single model tetap terbaik.

### 4. Hyperparameter Tuning Results

Nested CV berhasil mengidentifikasi parameter optimal untuk setiap algoritma:

| Algoritma | Best Parameters |
|-----------|-----------------|
| XGBoost | max_depth=6, learning_rate=0.1, n_estimators=150 |
| Random Forest | max_depth=10, min_samples_leaf=2, n_estimators=150 |
| SVM | C=1.0, kernel=rbf, gamma=scale |
| RBF Network | n_centers=15, spread_factor=1.2 |
| Self-Training | threshold=0.75 |

## B. Pembahasan

1. **Strategi Imputasi:** Median Imputation terbukti paling efektif (F1: 0.6884) karena mempertahankan central tendency sambil robust terhadap outlier.

2. **Perbandingan Algoritma:** XGBoost mengungguli semua algoritma lain dengan margin signifikan. Self-Training menunjukkan potensi dengan memanfaatkan pola semi-supervised.

3. **Voting Ensemble:** Tidak memberikan peningkatan karena XGBoost sudah dominan dan variasi antar model tidak cukup besar untuk menghasilkan "wisdom of the crowd" effect.

4. **Keterbatasan:** Dengan hanya 3 fitur dan 230 sampel, ceiling performa model terbatas. Penambahan fitur dapat meningkatkan hasil.

---

# BAB V KESIMPULAN DAN SARAN

## A. Kesimpulan

1. Aktivitas mahasiswa dalam ELS dapat dipetakan ke dimensi FSLSM menggunakan pendekatan multi-label classification dengan memanfaatkan data waktu interaksi.

2. Strategi imputasi **Median** memberikan hasil terbaik (F1-Macro 0.6884) dibandingkan Mean, Zero, dan MICE.

3. **XGBoost** mencapai F1-Macro **0.7008**, menjadi algoritma terbaik di antara 5 algoritma yang diuji.

4. **Voting Ensemble** tidak meningkatkan performa dibandingkan XGBoost single model, menunjukkan bahwa dalam kasus ini pendekatan ensemble tidak diperlukan.

5. Lima algoritma berhasil dievaluasi dengan **Nested Cross-Validation** untuk hyperparameter tuning yang unbiased.

## B. Saran

1. **Feature Engineering:** Eksplorasi fitur tambahan seperti rasio waktu antar materi, total engagement time, dan pola temporal.

2. **Penambahan Fitur:** Mengintegrasikan fitur dari 6 kolom asli (termasuk tasks, forums, quizzes) jika data tersedia.

3. **Implementasi Sistem:** Mengintegrasikan model XGBoost ke dalam ELS Amikom untuk personalisasi pembelajaran real-time.

4. **Eksplorasi Algoritma Lanjutan:** Mencoba Classifier Chains atau Label Powerset untuk menangkap dependensi antar label.

---

# DAFTAR PUSTAKA

Baihaqi, I., et al. (2024). Framework Prediksi Gaya Belajar berbasis FSLSM dengan Integrasi Feature Selection Random Forest.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of KDD.

Chen, Y., et al. (2023). XGBoost for Multi-label Classification in Educational Data Mining.

Felder, R. M., & Silverman, L. K. (1988). Learning and Teaching Styles in Engineering Education.

Hasibuan, M. S. (2025). Penerapan FSLSM dalam Analitik Pembelajaran.

Mian, M., et al. (2022). Learning Analytics in Higher Education: A Systematic Review.

Sayed, A., et al. (2024). Predicting Student Learning Styles using Machine Learning on OULAD Dataset.

Zhang, M. L., & Zhou, Z. H. (2014). A Review on Multi-Label Learning Algorithms. IEEE Transactions on Knowledge and Data Engineering.
