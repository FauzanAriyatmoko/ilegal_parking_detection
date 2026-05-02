import streamlit as st
import json
import cv2
import numpy as np
import os
import requests
import time
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Config Paths
VIDEO_JSON_PATH = "config/parking_ilegal/video_full.json"
ZONES_DIR = "config/parking_ilegal"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Dashboard Illegal Parking", layout="wide")
st.title("🚗 Sistem Deteksi Parkir Ilegal")

@st.cache_data
def load_cameras():
    try:
        with open(VIDEO_JSON_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Gagal memuat {VIDEO_JSON_PATH}: {e}")
        return []

cameras = load_cameras()
camera_opts = {cam["id"]: cam["url"] for cam in cameras}

# Sidebar
st.sidebar.header("Pengaturan")
selected_cam_id = st.sidebar.selectbox("Pilih Lokasi CCTV", list(camera_opts.keys()))
mode = st.sidebar.radio("Mode Aplikasi", ["Pengaturan Zona (Zoning)", "Live View Deteksi"])

selected_url = camera_opts.get(selected_cam_id)
zone_file = os.path.join(ZONES_DIR, f"zones_{selected_cam_id}.json")

if mode == "Pengaturan Zona (Zoning)":
    st.subheader(f"Pengaturan Zona Parkir Ilegal: {selected_cam_id}")
    st.write("Gambarkan area parkir ilegal menggunakan alat gambar (poligon) di bawah ini.")
    
    # Ambil 1 frame untuk background
    @st.cache_data(ttl=60)
    def get_bg_image(url):
        try:
            jpg_url = url.replace(".m3u8", ".jpg").replace(".html", ".jpg")
            req = requests.get(jpg_url, timeout=5)
            if req.status_code == 200:
                nparr = np.frombuffer(req.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.warning(f"Fallback JPG error: {e}")

        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    bg_img_array = get_bg_image(selected_url)
    
    if bg_img_array is not None:
        bg_image = Image.fromarray(bg_img_array)
        width, height = bg_image.size
        
        # Streamlit Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Warna merah transparan
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=bg_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="polygon",
            key="canvas",
        )

        # Proses hasil gambar
        if st.button("Simpan Zona"):
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                zones = []
                for obj in objects:
                    if obj["type"] == "path":
                        polygon = []
                        for cmd in obj["path"]:
                            if cmd[0] in ['M', 'L']:
                                norm_x = cmd[1] / width
                                norm_y = cmd[2] / height
                                polygon.append((norm_x, norm_y))
                        if len(polygon) >= 3:
                            zones.append(polygon)
                
                # Simpan ke JSON
                data = {
                    "video_info": {
                        "width": width,
                        "height": height,
                        "cctv_name": selected_cam_id
                    },
                    "zones": zones
                }
                os.makedirs(ZONES_DIR, exist_ok=True)
                with open(zone_file, "w") as f:
                    json.dump(data, f, indent=4)
                
                st.success(f"Berhasil menyimpan {len(zones)} zona ke {zone_file}!")
            else:
                st.warning("Belum ada zona yang digambar.")
    else:
        st.error("Tidak dapat mengambil frame dari URL RTSP CCTV ini. Pastikan link aktif.")

elif mode == "Live View Deteksi":
    st.subheader(f"Live View: {selected_cam_id}")
    st.write(f"Stream URL: `{selected_url}`")
    
    # Cek status dari API
    try:
        res = requests.get(f"{API_URL}/camera/{selected_cam_id}/status")
        is_running = res.json().get("is_running", False)
    except:
        st.error("Backend Server (API) tidak aktif! Pastikan `python3 backend/runner.py` berjalan.")
        is_running = False
        
    col1, col2 = st.columns([1, 5])
    
    with col1:
        if not is_running:
            if st.button("▶️ Mulai Deteksi"):
                try:
                    requests.post(f"{API_URL}/camera/{selected_cam_id}/start")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal mengirim perintah: {e}")
        else:
            if st.button("⏹️ Hentikan Deteksi"):
                try:
                    requests.post(f"{API_URL}/camera/{selected_cam_id}/stop")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal mengirim perintah: {e}")

    with col2:
        if is_running:
            st.success("Service Deteksi Sedang Berjalan.")
            # Menampilkan stream MJPEG dari Backend
            stream_url = f"{API_URL}/camera/{selected_cam_id}/stream"
            import streamlit.components.v1 as components
            components.html(f'<img src="{stream_url}" style="width: 100%; height: auto;">', height=600)
        else:
            st.info("Deteksi dihentikan. Klik Mulai Deteksi untuk menjalankan AI di backend.")
