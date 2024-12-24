import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Tambahkan latar belakang dengan CSS
image_base64 = get_base64_image("src/assets/Background.jpg")
background_css = f"""
<style>
body {{
    background-image: url('data:image/jpg;base64,{image_base64}');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}}
.sidebar .sidebar-content {{
    background: linear-gradient(to bottom, #6a11cb, #2575fc);
    color: #ffffff;
    border-radius: 10px;
    padding: 15px;
}}
button {{
    background: linear-gradient(to right, #ff6a00, #ee0979);
    color: white;
    border: none;
    padding: 12px 25px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
    border-radius: 30px;
    transition: transform 0.3s, box-shadow 0.3s;
}}
button:hover {{
    transform: scale(1.1);
    box-shadow: 0px 10px 15px rgba(0, 0, 0, 0.3);
}}
h1 {{
    font-size: 60px;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    animation: glow 2s infinite alternate;
}}
@keyframes glow {{
    from {{ text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); }}
    to {{ text-shadow: 2px 2px 20px #ffffff; }}
}}
.prediction-box {{
    background: linear-gradient(to right, #1f4037, #99f2c8);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    color: #ffffff;
    font-weight: bold;
    box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
}}
footer {{
    margin-top: 50px;
    text-align: center;
    color: #bdc3c7;
    font-size: 14px;
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Judul aplikasi dengan efek animasi
st.markdown('<h1 style="text-align: center;">🐮 Klasifikasi Citra Jenis Sapi 🐮</h1>', unsafe_allow_html=True)

# Penjelasan aplikasi dengan desain kotak informasi modern
st.markdown(
    """
    <div style="background: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 15px; margin-bottom: 20px; color: white;">
        <p style="text-align: justify; font-size: 18px;">
            Selamat datang di aplikasi klasifikasi citra sapi! Proyek ini dirancang untuk membantu peternak dan peneliti 
            mengidentifikasi jenis sapi secara akurat dan cepat. Cukup unggah gambar sapi, pilih model prediksi, dan lihat hasilnya dalam hitungan detik!
        </p>
        <p style="text-align: justify; font-size: 18px;">
            Aplikasi ini mendukung berbagai jenis sapi seperti <b>Brahman</b>, <b>Cholistani</b>, <b>Dhani</b>, dan lainnya. 
            Jelajahi teknologi pembelajaran mesin terbaru dalam pengelolaan peternakan!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Fungsi prediksi
def predict(uploaded_image, model_path):
    class_names = [
        "brahman",
        "cholistani",
        "dhani",
        "fresian",
        "kankarej",
        "sahiwal",
        "sibbi",
        "unidentified (mixed)"
    ]

    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model(model_path)

    output = model.predict(img)
    score = tf.nn.softmax(output[0])
    return class_names[np.argmax(score)], score.numpy()

# Sidebar dengan desain modern
st.sidebar.markdown(
    """
    <h3 style="text-align: center; color: #ffffff;">🔍 Pilih Model</h3>
    """,
    unsafe_allow_html=True
)
model_option = st.sidebar.radio(
    "Pilih model untuk prediksi:",
    ("🌟 InceptionV3", "🚀 MobileNetV2")
)

if model_option == "🌟 InceptionV3":
    model_path = Path(__file__).parent / "Model/Image/Model_InceptionV3/model.h5"
else:
    model_path = Path(__file__).parent / "Model/Image/Model_MobileNetV2/model.h5"

# Komponen file uploader
st.markdown(
    """
    <h4 style="text-align: center; color: #ffffff;">📤 Unggah Gambar Sapi</h4>
    """,
    unsafe_allow_html=True
)
uploads = st.file_uploader(
    "Unggah citra untuk mendapatkan hasil prediksi", 
    type=["png", "jpg"], 
    accept_multiple_files=True
)

# Tombol prediksi
if st.button("🚀 Predict"):
    if uploads:
        st.subheader("Hasil prediksi:")

        for upload in uploads:
            st.image(upload, caption=f"Citra yang diunggah: {upload.name}", use_container_width=True)

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                try:
                    label, scores = predict(upload, model_path)
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <strong>Image:</strong> {upload.name}<br>
                            <strong>Label:</strong> {label}<br>
                            <strong>Confidence:</strong> {scores.max() * 100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Tambahkan grafik batang untuk confidence level
                    st.bar_chart(scores)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")

# Footer aplikasi
st.markdown(
    """
    <footer>
        Dibuat dengan ❤️ oleh Hilmi Naufal Ramadhani | 2024
    </footer>
    """,
    unsafe_allow_html=True
)
