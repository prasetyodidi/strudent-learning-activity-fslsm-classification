# BAB IPENDAHULUAN 

A. Latar Belakang Masalah 

Perkembangan Learning Analytics dan Educational Data Mining (EDM) telah memberikan kontribusi besar dalam dunia pendidikan modern. Dengan memanfaatkan data interaksi mahasiswa dalam sistem pembelajaran elektronik, pendidik dapat memahami pola belajar, memprediksi hasil akademik, serta merancang strategi pembelajaran yang lebih personal (Mian et al., 2022). Penerapan analitik ini menjadi semakin relevan di era digital, di mana proses belajar tidak lagi terbatas pada ruang kelas fisik, tetapi juga berlangsung melalui platform Electronic Learning System (ELS).

Salah satu aspek penting dalam personalisasi pembelajaran adalah memahami gaya belajar mahasiswa. Model Felder-Silverman Learning Style Model (FSLSM) banyak digunakan untuk mengklasifikasikan gaya belajar berdasarkan beberapa dimensi, seperti aktif atau reflektif, sensitif atau intuitif, visual atau verbal, serta sequential atau global (Hasibuan, 2025). Dengan memahami preferensi belajar tersebut, pendidik dapat menyesuaikan materi dan metode penyampaian sehingga meningkatkan keterlibatan dan hasil belajar mahasiswa.

Penelitian sebelumnya telah mengkombinasikan FSLSM dengan berbagai algoritma machine learning klasik seperti Logistic Regression, K-Nearest Neighbor, Random Forest, dan Support Vector Machine (SVM) (Baihaqi, 2024). Hasilnya menunjukkan peningkatan akurasi dalam prediksi gaya belajar, terutama ketika disertai teknik feature selection dan penanganan data imbalance. Namun, pendekatan tersebut masih memiliki keterbatasan dalam menangani data sekuensial, khususnya data interaksi mahasiswa yang bersifat berurutan dalam platform e-learning (Chauhan et al., 2022).

Dalam konteks Universitas Amikom Purwokerto, data log aktivitas mahasiswa pada Electronic Learning System (ELS) menjadi sumber yang sangat potensial untuk dianalisis. Pola klik, akses materi, forum diskusi, hingga penyelesaian tugas dapat memberikan gambaran yang kaya mengenai gaya belajar mahasiswa. Oleh karena itu, diperlukan pendekatan yang mampu menangkap dinamika temporal dari data tersebut.

Deep Learning, khususnya algoritma Long Short-Term Memory (LSTM), menawarkan keunggulan dalam memproses data sekuensial karena kemampuannya mempertahankan informasi jangka panjang (Hochreiter & Schmidhuber, 1997; Kalita et al., 2025). Dengan mengkombinasikan LSTM dan FSLSM, penelitian ini berupaya membangun model prediksi gaya belajar mahasiswa yang lebih akurat. Model ini diharapkan tidak hanya memberikan kontribusi pada pengembangan teori di bidang Learning Analytics, tetapi juga memberikan manfaat praktis berupa personalisasi pembelajaran di lingkungan Amikom Purwokerto.

B. Rumusan Masalah 

1. Bagaimana memetakan aktivitas mahasiswa dalam ELS ke dimensi FSLSM? 


2. Bagaimana membangun model prediksi gaya belajar menggunakan LSTM? 


3. Seberapa baik performa LSTM dibandingkan model pembanding (seperti SVM atau ANN klasik)? 



C. Batasan Masalah 

1. Dataset terfokus pada ELS Amikom Purwokerto. 


2. Model deep learning hanya pada LSTM. 


3. Gaya belajar mengacu pada dimensi FSLSM. 



D. Tujuan Penelitian 

1. Menerapkan FSLSM dalam analisis gaya belajar berbasis data ELS. 


2. Mengembangkan model prediksi dengan LSTM. 


3. Mengevaluasi model dengan metrik standar evaluasi ML. 



E. Manfaat Penelitian 

1. **Teoritis:** kontribusi pada penelitian Learning Analytics berbasis deep learning. 


2. **Praktis:** mendukung personalisasi pembelajaran di Amikom Purwokerto. 



---

# BAB IITINJAUAN PUSTAKA 

A. Landasan Teori 

1. Learning Analytics dan Educational Data Mining (EDM) 

Learning Analytics adalah proses pengumpulan, analisis, dan pelaporan data mengenai mahasiswa dan konteks pembelajaran untuk memahami serta mengoptimalkan proses belajar. Sementara itu, Educational Data Mining (EDM) fokus pada penerapan teknik data mining untuk mengeksplorasi pola tersembunyi dalam data pendidikan. Kedua bidang ini semakin penting seiring meningkatnya penggunaan Learning Management System (LMS) dan Electronic Learning System (ELS) di perguruan tinggi.

2. Electronic Learning System (ELS) 

ELS merupakan sistem yang digunakan untuk mendukung pembelajaran daring di Universitas Amikom Purwokerto. Sistem ini mencatat berbagai aktivitas mahasiswa, seperti akses materi, forum diskusi, pengumpulan tugas, hingga capaian nilai. Data interaksi ini dapat dimanfaatkan sebagai basis analisis gaya belajar mahasiswa.

3. Gaya Belajar dan Felder-Silverman Learning Style Model (FSLSM) 

Model FSLSM yang dikembangkan oleh Felder dan Silverman (1988) mengklasifikasikan gaya belajar mahasiswa ke dalam empat dimensi:

- **Processing:** aktif → reflektif 


- **Perception:** sensitif → intuitif 


- **Input:** visual → verbal 


- **Understanding:** sequential → global 



Penerapan FSLSM dalam analitik pembelajaran memungkinkan dosen memahami preferensi belajar mahasiswa dan menyesuaikan strategi pengajaran agar lebih efektif .

4. Machine Learning dalam Analisis Gaya Belajar 

Berbagai algoritma machine learning seperti Logistic Regression, K-Nearest Neighbor (KNN), Random Forest, dan Support Vector Machine (SVM) telah digunakan dalam penelitian prediksi gaya belajar. Algoritma tersebut mampu mengidentifikasi pola dari data interaksi mahasiswa, namun memiliki keterbatasan ketika berhadapan dengan data sekuensial yang kompleks.

5. Deep Learning dan Long Short-Term Memory (LSTM) 

Deep Learning adalah cabang machine learning yang menggunakan jaringan saraf tiruan berlapis untuk menangkap representasi kompleks dari data. Salah satu arsitektur populer adalah Long Short-Term Memory (LSTM), yang dirancang untuk mengatasi kelemahan Recurrent Neural Network (RNN) dalam mempertahankan informasi jangka panjang. LSTM sangat efektif dalam memproses data sekuensial, sehingga sesuai digunakan untuk menganalisis log interaksi mahasiswa pada ELS.

B. Penelitian Sebelumnya 

1. Baihaqi dkk. (2024) mengembangkan framework prediksi gaya belajar berbasis FSLSM dengan integrasi feature selection Random Forest dan penggunaan Artificial Neural Network (ANN). Hasil penelitian menunjukkan peningkatan akurasi hingga 97%, namun masih menghadapi tantangan overfitting dan keterbatasan pada data temporal.


2. Sayed dkk. (2024) meneliti prediksi gaya belajar mahasiswa menggunakan dataset OULAD dengan algoritma Logistic Regression, KNN, Random Forest, dan SVM. Penelitian ini menunjukkan bahwa metode machine learning mampu meningkatkan akurasi prediksi, namun hanya terbatas pada data tabular dan belum optimal untuk data sekuensial.


3. Kalita et al. (2025) dalam artikel "Predicting student academic performance using Bi-LSTM" menemukan bahwa model Bi-LSTM outperform model machine learning tradisional dalam memprediksi performa akademik mahasiswa dengan akurasi 88% dan presisi 92%, menegaskan keunggulan LSTM dalam menangkap pola sekuensial dari data akademik untuk intervensi dini.



---

# BAB III METODE PENELITIAN 

A. Tempat dan Waktu Penelitian (Bila Ada) 

Penelitian ini dilakukan di Universitas Amikom Purwokerto, dengan objek penelitian berupa data log aktivitas mahasiswa yang diperoleh dari Electronic Learning System (ELS). Dataset yang digunakan mencakup periode Agustus sampai Oktober 2025, yang memuat interaksi mahasiswa terhadap materi pembelajaran, forum diskusi, pengumpulan tugas, quiz, serta catatan nilai. Waktu penelitian dilaksanakan selama enam bulan, yang mencakup tahapan pengumpulan data, preprocessing, pembangunan model, pengujian, serta penulisan laporan akhir.

B. Metode Pengumpulan Data 

Metode pengumpulan data dilakukan dengan beberapa cara, yaitu:

1. **Observasi Sistem** Pengamatan langsung pada ELS Amikom Purwokerto untuk memahami struktur database, jenis aktivitas yang tercatat, serta pola interaksi mahasiswa.


2. **Ekstraksi Data Log** 
Data diperoleh dengan mengekstrak log aktivitas dari basis data ELS Amikom. Aktivitas yang dicatat meliputi akses ke materi kuliah, forum diskusi, quiz, dan tugas.


3. **Dokumentasi** Mengumpulkan dokumen pendukung terkait penggunaan ELS, termasuk informasi akademik mahasiswa (misalnya gender, program studi, serta capaian nilai akhir) untuk melengkapi proses analisis .



C. Alat dan Bahan Penelitian 

1. **Perangkat Keras (Hardware)** Penelitian ini menggunakan laptop pribadi dengan spesifikasi sebagai berikut:
    - Laptop: Lenovo Ideapad Slim 3 
    - Prosesor: AMD Ryzen 3 
    - RAM: 12 GB 
    - Storage: 512 GB SSD 


2. **Perangkat Lunak (Software)** 
    - Sistem Operasi: Linux 
    - Bahasa Pemrograman: Python 
    - Framework & Library: TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib 
    - Database: MySQL 


3. **Dataset** 
    - Sumber: Electronic Learning System (ELS) Amikom Purwokerto 
    - Periode: Agustus-Oktober 2025 
    - Bentuk data: log interaksi mahasiswa (klik pada materi, forum, tugas, quiz) serta informasi akademik mahasiswa.


D. Konsep Penelitian 

1. **Preprocessing Data** 

    - a. Pembersihan data (menghapus duplikasi, mengatasi nilai kosong, dan noise).

    - b. Label encoding untuk mengubah data kategorikal menjadi numerik.

    - c. Mapping aktivitas mahasiswa ke dimensi Felder-Silverman Learning Style Model (FSLSM): aktif/reflektif, sensitif/intuitif, visual/verbal, sequential/global.

    - d. Normalisasi data numerik untuk meningkatkan kinerja model.

    - e. Pembentukan sequence data dari log aktivitas untuk menjadi input pada model LSTM.




2. 
**Pemodelan Deep Learning (LSTM)** 

* a. Input: sequence aktivitas mahasiswa.


* b. Arsitektur model:
    1. Input Layer (dimensi sequence log aktivitas) 
    2. Hidden Layer (LSTM cells) 
    3. Dense Layer (fully connected) 
    4. Output Layer (kelas FSLSM) 

* c. Hyperparameter tuning: jumlah neuron, batch size, learning rate, jumlah epoch.




3. **Evaluasi Model** 


* a. Pembagian data: training (80%) dan testing (20%).


* b. Metrik evaluasi: Accuracy, Precision, Recall, dan F1-Score.


* c. Perbandingan hasil LSTM dengan baseline model (misalnya SVM atau ANN klasik).




4. **Analisis Hasil** 


* a. Mengevaluasi performa LSTM dalam memprediksi gaya belajar.


* b. Membandingkan hasil dengan baseline untuk menilai keunggulan model deep learning.


* c. Menyajikan implikasi praktis hasil penelitian terhadap personalisasi pembelajaran di Amikom Purwokerto.