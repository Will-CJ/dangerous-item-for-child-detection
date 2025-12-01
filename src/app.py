import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ================================
# CSS RESPONSIVE UNTUK MOBILE
# ================================
st.markdown("""
<style>
/* Biar layout enak di mobile */
@media (max-width: 768px) {
    .stMarkdown, .stTitle, h3, h2, h1, p, div {
        font-size: 14px !important;
        line-height: 1.25em !important;
        word-wrap: break-word !important;
    }
    .risk-box {
        font-size: 14px !important;
        padding: 12px !important;
    }
}
.risk-box {
    background: #ffd400;
    padding: 15px;
    border-left: 6px solid red;
    border-radius: 8px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ================================
# CONFIG
# ================================
MODEL_PATH = "models/mobilenetv2_classifier_after_augment3.h5"
IMG_WIDTH = 224
IMG_HEIGHT = 224

LABELS = ['Baterai', 'Bola', 'Gunting', 'Korek Api', 'Lego']

RISKS = {
    "Baterai": "Bahaya Kimia dan Tersedak. Baterai, terutama jenis kancing, sangat berbahaya jika tertelan. Selain risiko tersedak, baterai dapat terjepit di kerongkongan dan melepaskan zat kimia alkali yang dapat menyebabkan luka bakar internal yang serius dalam waktu singkat, mengakibatkan kerusakan jaringan permanen.",
    "Bola": "Bahaya Tersedak dan Pernapasan. Risiko utama datang dari bola dengan diameter kecil yang mudah masuk ke mulut anak. Jika tertelan, bola dapat menyumbat saluran pernapasan (trakea) sepenuhnya, menyebabkan tersedak dan henti napas dalam hitungan menit. Ini merupakan salah satu penyebab utama kecelakaan tersedak pada balita.",
    "Gunting": "Bahaya Fisik dan Cedera Luka. Gunting, baik yang tajam maupun tumpul, memiliki potensi menyebabkan luka robek atau sayatan pada kulit. Anak-anak dapat melukai diri sendiri, menusuk mata, atau mencederai orang lain. Gunting juga dapat menyebabkan cedera tumpul jika digunakan untuk memukul.",
    "Korek Api": "Bahaya Kebakaran dan Luka Bakar Parah. Korek api merupakan sumber bahaya yang sangat tinggi karena dapat memicu kebakaran dalam hitungan detik. Selain risiko kebakaran rumah, anak dapat mengalami luka bakar serius pada kulit, terutama di tangan dan wajah, jika mereka mencoba menyalakan atau bermain dengan api. Gas di dalamnya juga berbahaya jika terhirup.",
    "Lego": "Bahaya Tersedak dan Penyumbatan Saluran Cerna. Potongan Lego kecil sangat mudah tertelan oleh balita. Bahaya utamanya adalah tersedak jika tersangkut di kerongkongan, atau penyumbatan usus jika balok tersebut berhasil melewati lambung namun tersangkut di saluran pencernaan bagian bawah, yang memerlukan intervensi medis."
}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()


# ================================
# PREDICTION FUNCTION
# ================================
def predict_single_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
    arr = img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)
    idx = np.argmax(pred)
    return LABELS[idx], float(np.max(pred)) * 100


# ==========================================
# VIDEO PROCESSOR
# ==========================================
class DangerProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_time = 0
        self.interval = 1
        self.callback = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_time >= self.interval:
            label, conf = predict_single_image(img)
            self.last_time = now

            if self.callback:
                self.callback({"label": label, "conf": conf})

        # Overlay text
        lbl = st.session_state.get("live_label", "...")
        c = st.session_state.get("live_conf", 0.0)
        text = f"{lbl} ({c:.1f}%)"

        cv2.putText(img, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")


# ==========================================
# UI
# ==========================================
st.title("Deteksi Benda Berbahaya 👶")

mode = st.radio("Mode:", ["📸 Capture Photo", "🎥 Live Stream"], horizontal=True)
st.markdown("---")

# ================================
# MODE FOTO
# ================================
if mode == "📸 Capture Photo":
    img = st.camera_input("Ambil gambar")

    if img:
        frame = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        st.image(frame, channels="BGR", use_container_width=True)

        label, conf = predict_single_image(frame)
        st.subheader(f"Hasil: {label} ({conf:.2f}%)")

        st.markdown(f"""
        <div class='risk-box'>
            <b>⚠ Risiko:</b><br>{RISKS[label]}
        </div>
        """, unsafe_allow_html=True)


# ================================
# MODE LIVE STREAM
# ================================
else:
    st.subheader("Live Stream")

    if "live_label" not in st.session_state:
        st.session_state["live_label"] = "Menunggu..."
        st.session_state["live_conf"] = 0.0

    def update_result(result):
        st.session_state["live_label"] = result["label"]
        st.session_state["live_conf"] = result["conf"]
        st.session_state["updated"] = True

    webrtc_ctx = webrtc_streamer(
        key="live_mobile",
        video_processor_factory=DangerProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.callback = update_result

    placeholder = st.empty()

    if st.session_state.get("updated", False):
        st.session_state["updated"] = False
        label = st.session_state["live_label"]
        conf = st.session_state["live_conf"]

        placeholder.markdown(f"""
        <h3>{label}</h3>
        Akurasi: {conf:.2f}%<br>

        <div class='risk-box'>
            <b>⚠ Risiko:</b><br>{RISKS.get(label, "-")}
        </div>
        """, unsafe_allow_html=True)
