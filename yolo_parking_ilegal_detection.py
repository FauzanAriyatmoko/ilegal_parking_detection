import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from collections import defaultdict
import argparse # Import argparse

# Fungsi untuk memuat sumber CCTV dari file JSON
def load_cctv_sources(file_path='cctv_sources.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan.")
        return {}
    except Exception as e:
        print(f"Error saat memuat sumber CCTV: {e}")
        return {}

class IllegalParkingDetector:
    def __init__(self, model_path):
        """Inisialisasi detector"""
        self.model = YOLO(model_path)
        
        # Zona parkir ilegal (koordinat ternormalisasi 0-1)
        self.illegal_zones = []
        
        # Tracking mobil di zona ilegal
        # {vehicle_id: {'entry_time': float, 'class': str, 'zone_id': int}}
        self.vehicles_in_zone = {}
        
        # Kelas kendaraan yang dipantau. Untuk yolov8x.pt, 'car' adalah kelas 2.
        self.vehicle_classes = ['car']
        # Indeks kelas yang sesuai
        self.vehicle_class_indices = [2] # Hanya deteksi 'car'
        
        # Counter untuk ID unik kendaraan yang dilacak
        self.next_vehicle_id = 0
        
        # Threshold jarak untuk re-identifikasi kendaraan (dalam koordinat ternormalisasi)
        self.distance_threshold = 0.08
        
        # Kamus untuk menyimpan posisi terakhir kendaraan yang dilacak
        self.tracked_vehicles = {}

    def add_zone(self, points_normalized):
        """
        Tambah zona parkir ilegal.
        points_normalized: list of tuples [(x1, y1), (x2, y2), ...]
        """
        self.illegal_zones.append(np.array(points_normalized, dtype=np.float32))
        print(f"Zona {len(self.illegal_zones)} ditambahkan dengan {len(points_normalized)} titik.")

    def denormalize_zone(self, zone_norm, frame_shape):
        """Konversi zona dari koordinat ternormalisasi ke koordinat piksel."""
        height, width, _ = frame_shape
        return np.array([ (int(pt[0] * width), int(pt[1] * height)) for pt in zone_norm ], dtype=np.int32)

    def point_in_polygon(self, point, polygon):
        """Cek apakah sebuah titik berada di dalam poligon."""
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def get_box_bottom_center(self, box):
        """Dapatkan titik tengah bawah dari bounding box untuk posisi kendaraan."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y2)

    def find_existing_vehicle(self, center_norm):
        """Cari kendaraan yang sudah ada yang paling dekat dengan deteksi baru."""
        nearest_id = None
        min_dist = self.distance_threshold
        
        for vehicle_id, data in self.tracked_vehicles.items():
            dist = np.linalg.norm(np.array(center_norm) - np.array(data['center']))
            if dist < min_dist:
                min_dist = dist
                nearest_id = vehicle_id
        
        return nearest_id

    def format_duration(self, seconds):
        """Format durasi dalam detik ke format MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def draw_zones(self, frame):
        """Gambar zona parkir ilegal pada frame."""
        overlay = frame.copy()
        for i, zone_norm in enumerate(self.illegal_zones):
            zone_pixels = self.denormalize_zone(zone_norm, frame.shape)
            cv2.fillPoly(overlay, [zone_pixels], (0, 0, 255))
            cv2.polylines(frame, [zone_pixels], True, (0, 0, 255), 2)
            
            # Label zona
            centroid = np.mean(zone_pixels, axis=0).astype(int)
            cv2.putText(frame, f"ZONA ILEGAL {i+1}", (centroid[0] - 50, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame

    def process_video(self, video_path, output_path, show_preview=True, violation_threshold=2):
        """Proses video untuk deteksi parkir ilegal."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Tidak bisa membuka video di {video_path}")
            return

        # Properti video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        
        if show_preview:
            cv2.namedWindow("Deteksi Parkir Ilegal", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time_sec = frame_count / fps
            
            # Deteksi kendaraan
            results = self.model(frame, classes=self.vehicle_class_indices, conf=0.5)
            
            current_frame_detections = []
            
            # Normalisasi resolusi frame untuk konsistensi
            frame_height, frame_width, _ = frame.shape
            
            # Reset kendaraan yang terlacak di frame ini
            current_frame_tracked_ids = set()

            for det in results[0].boxes:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = det.conf[0].cpu().numpy()
                cls = int(det.cls[0].cpu().numpy())
                class_name = self.model.names[cls]

                # Dapatkan titik tengah bawah (posisi roda) dan normalisasi
                center_pixel = self.get_box_bottom_center([x1, y1, x2, y2])
                center_norm = (center_pixel[0] / frame_width, center_pixel[1] / frame_height)
                
                # Cek apakah kendaraan berada di dalam zona ilegal
                in_illegal_zone = False
                zone_idx = -1
                for i, zone_norm in enumerate(self.illegal_zones):
                    if self.point_in_polygon(center_norm, zone_norm):
                        in_illegal_zone = True
                        zone_idx = i
                        break
                
                # Tracking kendaraan
                vehicle_id = self.find_existing_vehicle(center_norm)
                if vehicle_id is None:
                    vehicle_id = self.next_vehicle_id
                    self.next_vehicle_id += 1
                
                self.tracked_vehicles[vehicle_id] = {'center': center_norm, 'last_seen': frame_count}
                current_frame_tracked_ids.add(vehicle_id)

                # Proses logika parkir ilegal
                if in_illegal_zone:
                    if vehicle_id not in self.vehicles_in_zone:
                        # Kendaraan baru masuk zona ilegal
                        self.vehicles_in_zone[vehicle_id] = {
                            'entry_time': current_time_sec,
                            'class': class_name,
                            'zone_id': zone_idx
                        }
                    
                    # Hitung durasi
                    duration = current_time_sec - self.vehicles_in_zone[vehicle_id]['entry_time']
                    
                    # Tentukan warna bounding box
                    is_violation = duration >= violation_threshold
                    color = (0, 0, 255) if is_violation else (0, 165, 255) # Merah jika pelanggaran, Oranye jika peringatan
                    
                    # Gambar BBox dan label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} | ID: {vehicle_id} | {self.format_duration(duration)}"
                    if is_violation:
                            label = f"{class_name} | PELANGGARAN: {self.format_duration(duration)}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                else:
                    # Kendaraan di luar zona, hapus dari daftar pelanggar jika ada
                    if vehicle_id in self.vehicles_in_zone:
                        del self.vehicles_in_zone[vehicle_id]
                    
                    # Gambar BBox normal
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Hapus kendaraan yang sudah lama tidak terlihat
            stale_ids = [vid for vid, data in self.tracked_vehicles.items() if (frame_count - data['last_seen']) > fps] # Hilang selama 1 detik
            for vid in stale_ids:
                if vid in self.tracked_vehicles: del self.tracked_vehicles[vid]
                if vid in self.vehicles_in_zone: del self.vehicles_in_zone[vid]

            # Gambar zona pada frame
            frame = self.draw_zones(frame)
            
            # Tampilkan info tambahan
            cv2.putText(frame, f"Kendaraan di Zona Ilegal: {len(self.vehicles_in_zone)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Tulis frame ke video output
            out.write(frame)

            # Tampilkan preview jika diaktifkan
            if show_preview:
                cv2.imshow("Deteksi Parkir Ilegal", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Lepaskan resource
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Proses selesai. Video hasil disimpan di: {output_path}")

# ============================================ 
# PENGGUNAAN UTAMA
# ============================================ 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistem Deteksi Parkir Ilegal dengan Zoning (YOLOv8)")
    parser.add_argument('cctv_name', type=str, 
                        help="Nama CCTV dari 'cctv_sources.json' (misal: 'btm_kota_bogor')")
    args = parser.parse_args()

    cctv_sources = load_cctv_sources()
    
    if args.cctv_name not in cctv_sources:
        print(f"Error: Nama CCTV '{args.cctv_name}' tidak ditemukan di cctv_sources.json.")
        print("Nama CCTV yang tersedia:")
        for name in cctv_sources.keys():
            print(f"- {name}")
        exit()
    
    video_path = cctv_sources[args.cctv_name]

    # 1. Inisialisasi detector dengan model yolov8s
    detector = IllegalParkingDetector(model_path='model/yolov8s.pt')
    
    # 2. Muat zona dari file JSON
    zones_file = f'hasil_zoning/parking_zones_{args.cctv_name}.json' # Dinamis
    try:
        with open(zones_file, 'r') as f:
            data = json.load(f)
        
        # Periksa apakah video_path dari JSON sesuai dengan yang diminta
        if data['video_info']['path'] != video_path:
            print(f"Peringatan: Zona parkir untuk '{args.cctv_name}' dibuat dengan video berbeda.")
            print(f"Path video di zona JSON: {data['video_info']['path']}")
            print(f"Path video yang digunakan: {video_path}")
            print("Pastikan zona ini relevan untuk video yang dipilih.")
            
        zones_from_json = data['zones']
        
        for zone in zones_from_json:
            # Konversi list of lists [x, y] menjadi list of tuples (x, y)
            zone_as_tuples = [tuple(point) for point in zone]
            detector.add_zone(zone_as_tuples)
            
    except FileNotFoundError:
        print(f"Error: File zona '{zones_file}' tidak ditemukan. Harap buat zona menggunakan zoning_tools.py terlebih dahulu!")
        exit()
    except Exception as e:
        print(f"Error saat memuat zona: {e}")
        exit()

    # 3. Siapkan path output yang unik
    output_dir = 'hasil'
    os.makedirs(output_dir, exist_ok=True)

    base_filename = f'deteksi_parkir_ilegal_yolo_{args.cctv_name}'
    counter = 1
    while True:
        output_filename = f"{base_filename}_{counter}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            break
        counter += 1

    print(f"Video output akan disimpan di: {output_path}")

    # 4. Proses video
    print("\nMemulai proses deteksi parkir ilegal...")
    print(f"Video sumber: {video_path}")
    print("Tekan 'q' pada jendela preview untuk menghentikan proses.")
    
    detector.process_video(
        video_path=video_path,
        output_path=output_path,
        show_preview=True,
        violation_threshold=20  # Dianggap pelanggaran setelah 20 detik
    )
