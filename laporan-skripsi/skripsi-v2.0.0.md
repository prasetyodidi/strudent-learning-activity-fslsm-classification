---
title: "Proposal Skripsi - Prediksi Gaya Belajar FSLSM dengan Multi-Label Classification"
version: "2.0.0"
date: "2025-12-14"
author: "Didi Prasetyo"
status: "Draft"
previous_version: "skripsi-v1.0.0.md"
changelog: "skripsi-v2.0.0-changelog.md"
---

# BAB I PENDAHULUAN

## A. Latar Belakang Masalah

Perkembangan Learning Analytics dan Educational Data Mining (EDM) telah memberikan kontribusi besar dalam dunia pendidikan modern. Dengan memanfaatkan data interaksi mahasiswa dalam sistem pembelajaran elektronik, pendidik dapat memahami pola belajar, memprediksi hasil akademik, serta merancang strategi pembelajaran yang lebih personal (Mian et al., 2022). Penerapan analitik ini menjadi semakin relevan di era digital, di mana proses belajar tidak lagi terbatas pada ruang kelas fisik, tetapi juga berlangsung melalui platform Electronic Learning System (ELS).

Salah satu aspek penting dalam personalisasi pembelajaran adalah memahami gaya belajar mahasiswa. Model Felder-Silverman Learning Style Model (FSLSM) banyak digunakan untuk mengklasifikasikan gaya belajar berdasarkan beberapa dimensi, seperti aktif atau reflektif, sensitif atau intuitif, visual atau verbal, serta sequential atau global (Hasibuan, 2025). Dengan memahami preferensi belajar tersebut, pendidik dapat menyesuaikan materi dan metode penyampaian sehingga meningkatkan keterlibatan dan hasil belajar mahasiswa.

Penelitian sebelumnya telah mengkombinasikan FSLSM dengan berbagai algoritma machine learning klasik seperti Logistic Regression, K-Nearest Neighbor, Random Forest, dan Support Vector Machine (SVM) (Baihaqi, 2024). Hasilnya menunjukkan peningkatan akurasi dalam prediksi gaya belajar, terutama ketika disertai teknik feature selection dan penanganan data imbalance. Namun, penelitian tersebut umumnya menggunakan pendekatan single-label classification, padahal dalam kenyataannya seorang mahasiswa dapat memiliki kombinasi beberapa gaya belajar sekaligus (multi-label).

Dalam konteks Universitas Amikom Purwokerto, data log aktivitas mahasiswa pada Electronic Learning System (ELS) menjadi sumber yang sangat potensial untuk dianalisis. Pola waktu akses materi video, dokumen, artikel, forum diskusi, hingga penyelesaian tugas dapat memberikan gambaran yang kaya mengenai gaya belajar mahasiswa. Oleh karena itu, diperlukan pendekatan **multi-label classification** yang mampu memprediksi kombinasi gaya belajar secara simultan.

Algoritma ensemble seperti **XGBoost** (Extreme Gradient Boosting) dan **Random Forest** telah terbukti efektif dalam berbagai tugas klasifikasi, termasuk multi-label classification (Chen et al., 2023; Zhang & Zhou, 2014). Dengan mengkombinasikan teknik penanganan missing value (imputasi), oversampling untuk mengatasi class imbalance, serta cross-validation yang ketat, penelitian ini berupaya membangun model prediksi gaya belajar mahasiswa yang andal. Model ini diharapkan tidak hanya memberikan kontribusi pada pengembangan teori di bidang Learning Analytics, tetapi juga memberikan manfaat praktis berupa personalisasi pembelajaran di lingkungan Amikom Purwokerto.

## B. Rumusan Masalah

1. Bagaimana memetakan aktivitas mahasiswa dalam ELS ke dimensi FSLSM menggunakan pendekatan multi-label classification?
2. Bagaimana pengaruh strategi imputasi missing value (Zero, Mean, Median, MICE) terhadap performa model prediksi gaya belajar?
3. Bagaimana membangun model prediksi gaya belajar menggunakan algoritma ensemble (XGBoost, Random Forest)?
4. Seberapa baik performa model ensemble dibandingkan dengan baseline (SVM)?

## C. Batasan Masalah

1. Dataset terfokus pada ELS Amikom Purwokerto dengan data waktu interaksi mahasiswa terhadap berbagai jenis materi pembelajaran.
2. Model machine learning yang digunakan meliputi XGBoost, Random Forest, dan SVM sebagai baseline.
3. Gaya belajar mengacu pada dua dimensi utama FSLSM: Processing (Aktif/Reflektif) dan Input (Visual/Verbal).
4. Pendekatan klasifikasi menggunakan multi-label classification dengan MultiOutputClassifier.

## D. Tujuan Penelitian

1. Menerapkan FSLSM dalam analisis gaya belajar berbasis data ELS dengan pendekatan multi-label.
2. Membandingkan efektivitas empat strategi imputasi missing value terhadap performa model.
3. Mengembangkan model prediksi gaya belajar dengan algoritma XGBoost dan Random Forest.
4. Mengevaluasi model dengan metrik standar multi-label classification (F1-Macro, Hamming Loss, Subset Accuracy).

## E. Manfaat Penelitian

1. **Teoritis:** Kontribusi pada penelitian Learning Analytics berbasis multi-label classification dan ensemble machine learning, serta pemahaman tentang pengaruh strategi imputasi pada performa model.
2. **Praktis:** Mendukung personalisasi pembelajaran di Amikom Purwokerto dengan menyediakan model prediksi gaya belajar yang dapat diintegrasikan ke dalam ELS.

---

# BAB II TINJAUAN PUSTAKA

## A. Landasan Teori

### 1. Learning Analytics dan Educational Data Mining (EDM)

Learning Analytics adalah proses pengumpulan, analisis, dan pelaporan data mengenai mahasiswa dan konteks pembelajaran untuk memahami serta mengoptimalkan proses belajar. Sementara itu, Educational Data Mining (EDM) fokus pada penerapan teknik data mining untuk mengeksplorasi pola tersembunyi dalam data pendidikan. Kedua bidang ini semakin penting seiring meningkatnya penggunaan Learning Management System (LMS) dan Electronic Learning System (ELS) di perguruan tinggi.

### 2. Electronic Learning System (ELS)

ELS merupakan sistem yang digunakan untuk mendukung pembelajaran daring di Universitas Amikom Purwokerto. Sistem ini mencatat berbagai aktivitas mahasiswa, termasuk waktu yang dihabiskan untuk mengakses materi video, dokumen, artikel, forum diskusi, dan kuis. Data interaksi berbasis waktu ini dapat dimanfaatkan sebagai basis analisis gaya belajar mahasiswa.

### 3. Gaya Belajar dan Felder-Silverman Learning Style Model (FSLSM)

Model FSLSM yang dikembangkan oleh Felder dan Silverman (1988) mengklasifikasikan gaya belajar mahasiswa ke dalam empat dimensi:

- **Processing:** Aktif → Reflektif
- **Perception:** Sensitif → Intuitif
- **Input:** Visual → Verbal
- **Understanding:** Sequential → Global

Penelitian ini berfokus pada dua dimensi yang dapat dipetakan dari data waktu interaksi dengan materi: **Processing** (Aktif/Reflektif) dan **Input** (Visual/Verbal). Seorang mahasiswa dapat memiliki kombinasi gaya belajar, misalnya *Aktif-Visual* atau *Reflektif-Verbal*, sehingga diperlukan pendekatan multi-label classification.

### 4. Multi-Label Classification

Berbeda dengan single-label classification di mana setiap instance hanya memiliki satu label, multi-label classification memungkinkan satu instance memiliki beberapa label sekaligus (Zhang & Zhou, 2014). Dalam konteks prediksi gaya belajar FSLSM, seorang mahasiswa dapat diklasifikasikan ke dalam beberapa gaya belajar bersamaan.

Metode umum untuk multi-label classification meliputi:
- **Binary Relevance:** Membangun classifier independen untuk setiap label.
- **Classifier Chains:** Memodelkan dependensi antar label dengan rantai classifier.
- **MultiOutputClassifier:** Wrapper scikit-learn yang menerapkan Binary Relevance.

### 5. Algoritma Ensemble: XGBoost dan Random Forest

**XGBoost (Extreme Gradient Boosting)** adalah algoritma ensemble berbasis gradient boosting yang telah terbukti efektif dalam berbagai kompetisi machine learning (Chen & Guestrin, 2016). Keunggulannya meliputi penanganan missing value, regularization untuk mencegah overfitting, dan efisiensi komputasi.

**Random Forest** adalah algoritma ensemble yang membangun banyak decision tree dan menggabungkan prediksinya melalui voting. Algoritma ini robust terhadap noise dan overfitting, serta mampu memberikan feature importance.

### 6. Penanganan Missing Value: Strategi Imputasi

Missing value merupakan tantangan umum dalam data analitik pembelajaran. Beberapa strategi imputasi yang digunakan:

| Strategi | Deskripsi | Karakteristik |
|----------|-----------|---------------|
| **Zero** | Mengisi missing value dengan 0 | Menganggap tidak ada aktivitas |
| **Mean** | Mengisi dengan rata-rata kolom | Asumsi aktivitas rata-rata |
| **Median** | Mengisi dengan median kolom | Robust terhadap outlier |
| **MICE** | Multiple Imputation by Chained Equations | Menangkap korelasi antar fitur |

### 7. Penanganan Class Imbalance: Random Oversampling

Class imbalance terjadi ketika distribusi label tidak seimbang. Random Oversampling adalah teknik untuk menyeimbangkan distribusi dengan menduplikasi sampel dari kelas minoritas. Dalam konteks multi-label, teknik ini diterapkan berdasarkan kombinasi label.

## B. Penelitian Sebelumnya

1. **Baihaqi dkk. (2024)** mengembangkan framework prediksi gaya belajar berbasis FSLSM dengan integrasi feature selection Random Forest dan penggunaan Artificial Neural Network (ANN). Hasil penelitian menunjukkan akurasi hingga 97%, namun masih menggunakan pendekatan single-label.

2. **Sayed dkk. (2024)** meneliti prediksi gaya belajar mahasiswa menggunakan dataset OULAD dengan algoritma Logistic Regression, KNN, Random Forest, dan SVM. Penelitian ini menggunakan data tabular dan pendekatan single-label classification.

3. **Chen et al. (2023)** menunjukkan bahwa XGBoost efektif untuk dataset berukuran kecil hingga menengah dengan performa yang konsisten bahkan dengan hyperparameter default.

4. **Zhang & Zhou (2014)** dalam review tentang multi-label learning algorithms memberikan fondasi teoretis untuk berbagai pendekatan multi-label classification yang dapat diterapkan dalam prediksi gaya belajar.

---

# BAB III METODE PENELITIAN

## A. Tempat dan Waktu Penelitian

Penelitian ini dilakukan di Universitas Amikom Purwokerto, dengan objek penelitian berupa data log aktivitas mahasiswa yang diperoleh dari Electronic Learning System (ELS). Dataset yang digunakan mencakup periode akademik 2024-2025, yang memuat data waktu interaksi mahasiswa terhadap berbagai jenis materi pembelajaran. Waktu penelitian dilaksanakan selama enam bulan, yang mencakup tahapan pengumpulan data, preprocessing, pembangunan model, pengujian, serta penulisan laporan akhir.

## B. Metode Pengumpulan Data

Metode pengumpulan data dilakukan dengan beberapa cara:

1. **Ekstraksi Data Log**
   Data diperoleh dengan mengekstrak log aktivitas dari basis data ELS Amikom. Data yang diekstrak meliputi waktu yang dihabiskan mahasiswa pada berbagai jenis materi: video, dokumen, artikel, tugas, forum, dan kuis.

2. **Dokumentasi FSLSM**
   Mengumpulkan hasil assessment FSLSM mahasiswa sebagai ground truth label untuk supervised learning.

3. **Integrasi Dataset**
   Menggabungkan data waktu interaksi dengan hasil assessment FSLSM berdasarkan ID mahasiswa (inner join).

## C. Alat dan Bahan Penelitian

### 1. Perangkat Keras (Hardware)
- Laptop: Lenovo Ideapad Slim 3
- Prosesor: AMD Ryzen 3
- RAM: 12 GB
- Storage: 512 GB SSD

### 2. Perangkat Lunak (Software)
- Sistem Operasi: Linux/macOS
- Bahasa Pemrograman: Python 3.10+
- Library: scikit-learn, XGBoost, imbalanced-learn, Pandas, NumPy, Matplotlib, Seaborn

### 3. Dataset
- **Sumber:** Electronic Learning System (ELS) Amikom Purwokerto
- **Jumlah Sampel Awal:** 604 mahasiswa (FSLSM assessment)
- **Jumlah Sampel Setelah Inner Join:** 123 mahasiswa
- **Jumlah Sampel Setelah Oversampling:** 230 mahasiswa
- **Jumlah Fitur:** 6 (waktu pada berbagai materi)
- **Jumlah Label:** 4 (Aktif, Reflektif, Visual, Verbal)

## D. Konsep Penelitian

### 1. Preprocessing Data

#### a. Pembersihan Data
- Penggabungan dataset FSLSM dengan data waktu interaksi (inner join).
- Identifikasi dan penanganan missing value (nilai 0 pada fitur waktu).

#### b. Strategi Imputasi Missing Value
Menerapkan empat strategi imputasi untuk perbandingan:
- Zero Imputation: Missing → 0
- Mean Imputation: Missing → Mean kolom
- Median Imputation: Missing → Median kolom
- MICE Imputation: Missing → Prediksi iteratif berdasarkan fitur lain

#### c. Label Encoding
Transformasi gaya belajar (Aktif, Reflektif, Visual, Verbal) ke format binary matrix untuk multi-label classification.

#### d. Normalisasi
Penerapan StandardScaler untuk normalisasi fitur numerik.

### 2. Penanganan Class Imbalance

#### a. Analisis Distribusi Label
Mengidentifikasi ketidakseimbangan distribusi kombinasi label:
- Majority: [Reflektif, Verbal] - 56.9%
- Minority: [Aktif, Visual] - 3.3%
- Imbalance Ratio: 17.5:1

#### b. Random Oversampling
Menerapkan Random Oversampling untuk menyeimbangkan distribusi kombinasi label, meningkatkan sampel dari 123 menjadi 230.

### 3. Pemodelan Machine Learning

#### a. Algoritma yang Digunakan
1. **XGBoost** dengan MultiOutputClassifier
2. **Random Forest** dengan MultiOutputClassifier
3. **SVM** dengan MultiOutputClassifier (baseline)

#### b. Hyperparameter Tuning
Menggunakan Nested Cross-Validation untuk optimasi hyperparameter:

**XGBoost:**
```
n_estimators: [100, 150, 200]
max_depth: [3, 6, 9]
learning_rate: [0.05, 0.1]
subsample: [0.8]
```

**Random Forest:**
```
n_estimators: [150]
max_depth: [10, 20, None]
min_samples_leaf: [2, 5]
max_features: ['sqrt']
```

### 4. Evaluasi Model

#### a. Metode Cross-Validation
- **Stratified K-Fold:** 10-fold dengan 3 repetisi
- **Monte Carlo CV:** 100 iterasi dengan 20% test size
- **Nested CV:** Cross-validation dalam cross-validation untuk hyperparameter tuning

#### b. Metrik Evaluasi
| Metrik | Deskripsi |
|--------|-----------|
| **F1-Macro** | Rata-rata F1-score per label (metrik utama) |
| **F1-Micro** | F1-score global berdasarkan total TP, FP, FN |
| **Hamming Loss** | Fraksi label yang salah diprediksi |
| **Subset Accuracy** | Persentase prediksi yang tepat semua label |
| **Precision-Macro** | Rata-rata precision per label |
| **Recall-Macro** | Rata-rata recall per label |

### 5. Analisis Hasil

#### a. Perbandingan Strategi Imputasi
Mengevaluasi F1-Macro dari masing-masing strategi imputasi untuk menentukan strategi terbaik.

#### b. Perbandingan Algoritma
Membandingkan performa XGBoost, Random Forest, dan SVM berdasarkan metrik evaluasi.

#### c. Analisis Feature Importance
Mengidentifikasi fitur yang paling berkontribusi terhadap prediksi gaya belajar.

#### d. Implikasi Praktis
Menyajikan rekomendasi penerapan model untuk personalisasi pembelajaran di ELS Amikom Purwokerto.

---

# BAB IV HASIL DAN PEMBAHASAN (Ringkasan)

## A. Hasil Eksperimen

### 1. Perbandingan Strategi Imputasi

Hasil perbandingan empat strategi imputasi menggunakan Random Forest:

| Strategi | F1-Macro | Std | Ranking |
|----------|----------|-----|---------|
| **Median** | **0.6884** | ±0.0834 | **1** |
| Mean | 0.6679 | ±0.0628 | 2 |
| Zero | 0.6413 | ±0.0686 | 3 |
| MICE | 0.6044 | ±0.0721 | 4 |

**Temuan:** Median Imputation memberikan hasil terbaik, kemungkinan karena robustness terhadap outlier pada data waktu interaksi.

### 2. Perbandingan Algoritma

Hasil perbandingan algoritma pada dataset terbaik (Median Imputation):

| Algoritma | F1-Macro | Status |
|-----------|----------|--------|
| **XGBoost** | **0.7006** | ✅ BEST |
| Random Forest | 0.6884 | ✅ Strong |
| SVM | 0.4311 | ⚠️ Baseline |

**Temuan:** XGBoost menunjukkan performa terbaik dengan F1-Macro 0.7006, mengalahkan Random Forest dan SVM secara signifikan.

### 3. Validasi dengan Benchmark

| Referensi | Expected | Achieved | Status |
|-----------|----------|----------|--------|
| Chen et al. (2023) - XGBoost | Strong for small data | 0.7006 F1-Macro | ✅ Validated |
| Benchmark F1-Macro | ≥ 0.65 | 0.7006 | ✅ Achieved |

## B. Pembahasan

1. **Strategi Imputasi:** Median Imputation terbukti paling efektif karena mempertahankan central tendency data waktu interaksi sambil robust terhadap outlier (mahasiswa dengan waktu belajar sangat panjang atau sangat pendek).

2. **Algoritma Ensemble:** XGBoost dan Random Forest secara signifikan mengungguli SVM dalam tugas multi-label classification ini. Hal ini menunjukkan bahwa algoritma ensemble lebih mampu menangkap pola kompleks dalam data.

3. **Multi-Label Approach:** Pendekatan multi-label memungkinkan prediksi kombinasi gaya belajar yang lebih realistis dibandingkan single-label classification.

---

# BAB V KESIMPULAN DAN SARAN

## A. Kesimpulan

1. Aktivitas mahasiswa dalam ELS dapat dipetakan ke dimensi FSLSM menggunakan pendekatan multi-label classification dengan memanfaatkan data waktu interaksi pada berbagai jenis materi pembelajaran.

2. Strategi imputasi Median memberikan hasil terbaik (F1-Macro 0.6884) dibandingkan Mean (0.6679), Zero (0.6413), dan MICE (0.6044).

3. Model XGBoost dengan MultiOutputClassifier mencapai F1-Macro 0.7006, menjadi algoritma terbaik untuk prediksi gaya belajar mahasiswa.

4. Model ensemble (XGBoost, Random Forest) secara signifikan mengungguli baseline SVM dalam tugas multi-label classification pada dataset ini.

## B. Saran

1. **Pengembangan Model:** Eksplorasi teknik multi-label lanjutan seperti Classifier Chains atau Label Powerset untuk menangkap dependensi antar label gaya belajar.

2. **Penambahan Fitur:** Mengintegrasikan fitur tambahan seperti pola temporal (jam akses), frekuensi login, dan interaksi forum untuk meningkatkan performa model.

3. **Implementasi Sistem:** Mengintegrasikan model prediksi ke dalam ELS Amikom untuk memberikan rekomendasi materi yang dipersonalisasi berdasarkan gaya belajar yang diprediksi.

4. **Validasi Eksternal:** Melakukan validasi model pada dataset dari institusi lain untuk menguji generalisasi model.

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
