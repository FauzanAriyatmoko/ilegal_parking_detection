# Sistem Deteksi Parkir Ilegal (Clean Architecture)

Repositori ini berisi sistem untuk mendeteksi pelanggaran parkir ilegal secara *real-time*. Sistem ini telah di-refactor menggunakan **Layer-First Clean Architecture** untuk mendukung pengembangan jangka panjang dan modularitas yang tinggi.

## Struktur Direktori

```text
Project Parking/
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
├── config/
│   ├── config.yaml               # Global: HTTP, MQTT, RTC, RTSP, log settings
│   └── parking_ilegal/
│       ├── config.yaml           # Parking-specific: model paths, confidence, interval
│       └── video.json            # List of RTSP camera URLs and IDs
├── backend/
│   ├── runner.py                 # Alternate entry point (main script run)
│   ├── app/
│   │   ├── main.py               # Main entry point: spawns HTTP server + AI manager
│   │   ├── ai_manager.py         # Loads config, starts ParkingDetector per camera
│   │   ├── http_server.py        # aiohttp HTTP + WebRTC server
│   │   └── rtsp_manager.py       # RTSP stream management utilities
│   └── internal/
│       ├── ai_runtime/
│       │   ├── background_process/
│       │   │   ├── periodic_frame.py       # PeriodicFrameCapture daemon thread
│       │   │   └── log_manager.py          # Log image storage & cleanup manager
│       │   └── parking_ilegal/
│       │       ├── parking_detect.py       # Core logic: vehicle detection & tracking in zone
│       │       └── zoning_tools.py         # Tool for creating zoning coordinates
│       ├── delivery/
│       │   ├── http/routes/                # HTTP route handlers
│       │   ├── mqtt/publisher.py           # MQTT publish client
│       │   └── rtc/rtc_track.py            # WebRTC video track
│       └── utils/
│           ├── geometry.py                 # Angle math helpers / normalization
│           ├── url_builder.py              # Build public image URL
│           └── utils.py                    # General image utilities
├── models/  
│   └── parking_ilegal/
│       └── yolov8s.pt                      # YOLOv8 model for vehicle detection
└── log/                                    # Log and snapshot storage directory
```

## Fitur Utama

-   **Layered Architecture**: Pemisahan jelas antara layer *Delivery* (HTTP/MQTT/RTC), *Application* (App logic, AI manager), dan *Internal* (AI runtime, core rules).
-   **Multi-Camera Support**: Dirancang untuk menangani banyak stream kamera secara paralel via `ai_manager.py` dan konfigurasi `video.json`.
-   **YOLOv8 & Tracking**: Deteksi objek dengan YOLOv8 dan ByteTrack untuk konsistensi ID.
-   **Zoning**: Definisi area parkir ilegal berbasis JSON.

## Instalasi

Pastikan Anda menggunakan Python 3.10 atau versi yang direkomendasikan.

```bash
pip install -r requirements.txt
```

Pastikan meletakkan model YOLOv8 (misal `yolov8s.pt`) ke dalam folder `models/parking_ilegal/`.

## Cara Menjalankan

### Konfigurasi

1. **Konfigurasi Global**: Atur port server, log level, dan MQTT di `config/config.yaml`.
2. **Kamera & Stream**: Daftarkan RTSP kamera di `config/parking_ilegal/video.json`.
3. **Zoning**: Gunakan `backend/internal/ai_runtime/parking_ilegal/zoning_tools.py` untuk menggambar zona per kamera, simpan hasilnya sesuai ID kamera (misal `config/parking_ilegal/zones_cctv_1.json`).

### Eksekusi

Anda dapat menjalankan backend dengan menggunakan script `runner.py`:

```bash
python backend/runner.py
```

Sistem akan otomatis:
- Memuat konfigurasi kamera.
- Memulai proses deteksi di setiap stream video.
- Menjalankan HTTP server / WebRTC (jika diaktifkan).
- Melakukan log dan menyimpan frame pelanggaran secara otomatis.
