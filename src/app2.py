import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase 

# ========================
# KONFIGURASI MODEL
# ========================
# Ganti sesuai nama file model Anda dan pastikan berada di folder 'models'
MODEL_PATH = "models/mobilenetv2_classifier_after_augment3.h5" 
IMG_HEIGHT = 224
IMG_WIDTH = 224

class_labels = ['Baterai', 'Bola', 'Gunting', 'Korek Api', 'Lego']

# Penjelasan risiko global 
RISKS = {
    "Baterai": "Berbahaya karena mengandung zat kimia dan risiko tertelan. Segera jauhkan!",
    "Bola": "Relatif aman, tapi bola kecil bisa menyebabkan tersedak. Perhatikan ukurannya.",
    "Gunting": "Berbahaya karena memiliki bagian tajam yang dapat melukai. Harus disimpan di tempat aman.",
    "Korek Api": "Sangat berbahaya karena dapat memicu kebakaran. Jauhkan dari jangkauan anak!",
    "Lego": "Risiko tersedak tinggi pada anak usia kecil. Awasi saat bermain."
}

# ========================
# LOAD MODEL
# ========================
st.title("Deteksi Benda Berbahaya untuk Anak 👶🔍 (Real-time)")

# Gunakan cache untuk memastikan model hanya dimuat sekali
@st.cache_resource
def load_model_cached(path):
    st.write("Model sedang dimuat...")
    # Pastikan 'compile=False' jika model hanya digunakan untuk inferensi
    model = tf.keras.models.load_model(path, compile=False) 
    return model

# Panggil fungsi untuk memuat model
try:
    model = load_model_cached(MODEL_PATH)
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file model ada di path: {MODEL_PATH}")
    st.exception(e)
    # Jika model gagal, hapus 'model' dari locals agar kondisi pengecekan di bawah berjalan
    if 'model' in locals():
        del model 


# ========================
# VIDEO TRANSFORMER CLASS (Untuk memproses frame demi frame)
# ========================

class DangerObjectDetector(VideoTransformerBase):
    """
    Kelas ini memproses setiap frame video yang datang dari kamera.
    Hanya menjalankan prediksi setiap 0.5 detik untuk menghemat sumber daya.
    """
    def __init__(self, model, class_labels, img_width, img_height):
        self.model = model
        self.class_labels = class_labels
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        
        # --- ATRIBUT UNTUK FRAME SKIPPING/DELAY ---
        self.last_prediction_time = time.time()
        # Interval waktu minimum antara prediksi (maks 2x per detik)
        self.prediction_interval = 0.5  

    def transform(self, frame):
        # Konversi VideoFrame (pyav) ke format NumPy array (OpenCV BGR)
        img = frame.to_ndarray(format="bgr24")

        # Ambil label dan conf terakhir dari session state
        label = st.session_state.get("current_label", "Menunggu Input...")
        conf = st.session_state.get("current_conf", 0.0)
        
        # --- LOGIKA FRAME SKIPPING ---
        current_time = time.time()
        
        # Cek apakah sudah waktunya untuk melakukan prediksi baru
        if (current_time - self.last_prediction_time) >= self.prediction_interval:
            
            # --- Logika Prediksi BARU ---
            
            # 1. Konversi BGR → RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. Resize ke ukuran training
            img_resized = cv2.resize(img_rgb, (self.IMG_WIDTH, self.IMG_HEIGHT))

            # 3. Konversi → array & normalisasi
            img_array = img_to_array(img_resized) / 255.0

            # 4. Tambahkan batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            # 5. Prediksi (verbose=0 agar tidak spam di console)
            pred = self.model.predict(img_batch, verbose=0)
            idx = np.argmax(pred)
            label = self.class_labels[idx]
            conf = float(np.max(pred)) * 100
            
            # Perbarui waktu terakhir prediksi
            self.last_prediction_time = current_time
            
            # Simpan hasil prediksi ke Streamlit session state 
            st.session_state["current_label"] = label
            st.session_state["current_conf"] = conf


        # --- Logika Display di Frame ---
        
        # Teks yang akan ditampilkan pada video
        text = f"Deteksi: {label} ({conf:.2f}%)"
        
        # Warna teks (Putih untuk kontras)
        text_color = (255, 255, 255) 
        # Warna latar belakang/outline (Hitam untuk kontras yang lebih baik)
        outline_color = (0, 0, 0)
        
        # Gambar teks dengan outline hitam agar lebih jelas di berbagai latar belakang
        cv2.putText(
            img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, outline_color, 6, cv2.LINE_AA
        )
        cv2.putText(
            img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA
        )

        # Kembalikan frame yang telah dimodifikasi (dalam format BGR)
        return img


# ========================
# STREAMLIT WEBRTC STREAMER
# ========================
st.subheader("🎥 Deteksi Lewat Kamera (Real-time)")

# Inisialisasi session state untuk menyimpan status prediksi
if "current_label" not in st.session_state:
    st.session_state["current_label"] = "Menunggu Input..."
    st.session_state["current_conf"] = 0.0

# 1. DEKLARASI webrtc_ctx di scope ini (di luar blok if) untuk menghilangkan Pylance warning
webrtc_ctx = None

# Membuat kolom untuk menata layout
col1, col2 = st.columns([2, 1])

with col1:
    if 'model' in locals():
        # Memulai stream WebRTC
        webrtc_ctx = webrtc_streamer(
            key="object-detector",
            # Menggunakan lambda untuk membuat instance DangerObjectDetector
            video_transformer_factory=lambda: DangerObjectDetector(
                model=model, 
                class_labels=class_labels, 
                img_width=IMG_WIDTH, 
                img_height=IMG_HEIGHT
            ),
            # Konfigurasi RTC (penting agar WebRTC berfungsi)
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )
    else:
        st.error("Tidak dapat memulai kamera karena model gagal dimuat.")


with col2:
    st.markdown("---")
    st.markdown("### Hasil Prediksi Real-time")
    
    # Placeholder untuk memperbarui hasil prediksi di UI
    label_placeholder = st.empty()
    conf_placeholder = st.empty()
    risk_placeholder = st.empty()
    
    # Loop untuk memperbarui UI (berjalan di main thread)
    # Cek apakah webrtc_ctx sudah dibuat dan sedang berjalan
    if webrtc_ctx is not None and webrtc_ctx.state.playing:
        while webrtc_ctx.state.playing:
            # Mengambil data dari session state yang diupdate oleh VideoTransformer
            label = st.session_state["current_label"]
            conf = st.session_state["current_conf"]

            label_placeholder.markdown(f"**Objek Terdeteksi:** **{label}**")
            conf_placeholder.markdown(f"**Akurasi Model:** `{conf:.2f}%`")

            # Penjelasan risiko
            risk_info = RISKS.get(label, "Aman, atau tidak ada dalam kategori benda berbahaya.")
            
            if label in RISKS:
                risk_placeholder.error(f"⚠️ RISIKO: {risk_info}")
            elif label != "Menunggu Input...":
                # Hanya tampilkan success jika akurasi cukup tinggi dan bukan objek berbahaya
                if conf > 50: 
                    risk_placeholder.success("✅ Terdeteksi: Benda non-berbahaya, atau di luar kategori model.")
                else:
                    risk_placeholder.info("Hasil prediksi masih kurang yakin. Dekatkan objek.")
            else:
                risk_placeholder.info("Mulai streaming untuk deteksi...")
                
            # Jeda untuk mengurangi frekuensi refresh UI
            time.sleep(0.5) 

# Tampilkan status jika kamera tidak berjalan (Perbaikan untuk Pylance)
if webrtc_ctx is not None and webrtc_ctx.state is not None and not webrtc_ctx.state.playing:
    st.warning("Tekan tombol 'START' di atas untuk menyalakan kamera dan memulai deteksi.")