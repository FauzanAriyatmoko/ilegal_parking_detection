import json
import logging
import threading
import time
import cv2

from backend.internal.ai_runtime.parking_ilegal.parking_detect import ParkingDetector

logger = logging.getLogger(__name__)

class CameraProcess(threading.Thread):
    def __init__(self, camera_id, url):
        super().__init__()
        self.camera_id = camera_id
        self.url = url
        zone_file = f"config/parking_ilegal/zones_{self.camera_id}.json"
        
        self.detector = ParkingDetector(
            camera_id=self.camera_id,
            zones_path=zone_file
        )
        self.running = False
        try:
            with open('loading.jpg', 'rb') as f:
                self.current_frame = f.read()
        except:
            self.current_frame = b''

    def run(self):
        self.running = True
        logger.info(f"[{self.camera_id}] Starting camera stream processing...")
        
        while self.running:
            use_snapshot_fallback = False
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                if ".m3u8" in self.url or ".html" in self.url:
                    use_snapshot_fallback = True
                    logger.info(f"[{self.camera_id}] OpenCV gagal membuka m3u8, berpindah ke Mode Snapshot JPG...")
                else:
                    logger.warning(f"[{self.camera_id}] Cannot open stream {self.url}, retrying in 5s...")
                    for _ in range(5):
                        if not self.running: break
                        time.sleep(1)
                    continue

            logger.info(f"[{self.camera_id}] Stream terhubung/siap diproses.")
            
            if use_snapshot_fallback:
                jpg_url = self.url.replace(".m3u8", ".jpg").replace(".html", ".jpg")
                import requests
                import numpy as np
                while self.running:
                    try:
                        req = requests.get(jpg_url, timeout=5)
                        if req.status_code == 200:
                            nparr = np.frombuffer(req.content, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                annotated_frame = self.detector.process_frame(frame)
                                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                                if ret:
                                    self.current_frame = buffer.tobytes()
                        else:
                            logger.warning(f"[{self.camera_id}] Gagal mengambil snapshot, kode: {req.status_code}")
                    except Exception as e:
                        logger.warning(f"[{self.camera_id}] Error saat snapshot HTTP: {e}")
                    
                    # Refresh rate ~3 frame per detik untuk mode snapshot (lebih mulus)
                    time.sleep(0.3)
            else:
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"[{self.camera_id}] Failed to read frame, reconnecting...")
                        break 
                    
                    annotated_frame = self.detector.process_frame(frame)
                    
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        self.current_frame = buffer.tobytes()
                    
                    time.sleep(0.03)

                cap.release()
                
            if self.running:
                time.sleep(2)

        logger.info(f"[{self.camera_id}] Stopped camera stream.")

    def stop(self):
        self.running = False

    def get_frame(self):
        return self.current_frame


class AIManager:
    def __init__(self, config_path="config/parking_ilegal/video_full.json"):
        self.config_path = config_path
        self.cameras_config = {}
        self.active_processes = {}
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                cams = json.load(f)
                self.cameras_config = {c["id"]: c["url"] for c in cams}
        except Exception as e:
            logger.error(f"Failed to load video config: {e}")
            
    def start_camera(self, camera_id):
        self.load_config() # Refresh config in case of new cameras
        if camera_id in self.active_processes and self.active_processes[camera_id].running:
            return False # Already running
            
        url = self.cameras_config.get(camera_id)
        if not url:
            logger.error(f"Camera ID {camera_id} not found in config.")
            return False
            
        process = CameraProcess(camera_id, url)
        process.daemon = True
        process.start()
        self.active_processes[camera_id] = process
        return True
        
    def stop_camera(self, camera_id):
        if camera_id in self.active_processes:
            self.active_processes[camera_id].stop()
            self.active_processes[camera_id].join()
            del self.active_processes[camera_id]
            return True
        return False
        
    def is_running(self, camera_id):
        return camera_id in self.active_processes and self.active_processes[camera_id].running

    def generate_frames(self, camera_id):
        while True:
            if camera_id in self.active_processes:
                frame = self.active_processes[camera_id].get_frame()
                if frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)

    def stop_all(self):
        for cid in list(self.active_processes.keys()):
            self.stop_camera(cid)
