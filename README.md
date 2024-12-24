# ✨ Klasifikasi Otomatis Jenis Ternak Sapi dengan ResNet50 dan MobileNetV2 ✨

<div style="text-align: center;">
  <img src="src/assetsReadme/Aes_CommonKingFisher.jpg" alt="" width="700">
</div>

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan model pembelajaran mesin yang mampu mengklasifikasikan jenis sapi berdasarkan fitur-fitur tertentu. Proyek ini dirancang untuk membantu peternak, peneliti, dan pihak terkait dalam mengidentifikasi jenis sapi dengan lebih cepat dan akurat, sehingga dapat mendukung pengambilan keputusan dalam pengelolaan peternakan, pemuliaan, dan pemasaran ternak.

**Link Dataset yang digunakan:** [Cow Breed Classification Dataset](https://www.kaggle.com/datasets/zaidworks0508/cow-breed-classification-dataset).
Preprocessing yang digunakan antara lain Resizing, Normalization dan Augmentation.

Model yang digunakan: Pre Trained Model **_InceptionV3_** dan Pre Trained Model **_MobileNetV2_** dengan Architecture Model kurang lebih seperti gambar berikut.

**InceptionV3 Architecture** ✨

<img src="src/assetsReadme/1_resnet_arch.png" alt="InceptionV3 Architecture" width="700">

**MobileNetV2 Architecture** ✨

![image](./src/assetsReadme/2_mobilenetv2arch.png)

## Overview Dataset

Dataset yang digunakan adalah Dataset citra sapi. Dataset terdiri atas 2.167 data yang terbagi menjadi 70% sebagai Training Set, 20% sebagai Validation Set, dan 10% sebagai Testing Set, dimana pada setiap Set, terdapat 8 Label Class yaitu 'brahman', 'cholistani', 'dhani', 'fresian', 'kankarej', 'sahiwal', 'sibbi', 'unidentified (mixed)'.

## Langkah Instalasi

a. Model

1. Unduh Dataset yang akan digunakan
2. Run Script Settings To Extracted Dataset (Script.ipynb)
3. Run IPYNB
4. Save Kedua Model

b. Streamlit

1. Clone Project Ini
2. PDM : [Dokumentasi PDM](https://pdm-project.org/)
3. PDM init
4. PDM add tensorflow numpy streamlit
5. Run app.py melalui localhost

c. Deploy Streamlit Model

1. Upload All File Diluar Environtment PDM (.gitignore)
2. Deploy dan Bake melalui streamlit dengan terhubung dengan github[Dokumentasi Streamlit](https://docs.streamlit.io/)
3. Run dengan address deployment dengan akhiran **.io**
