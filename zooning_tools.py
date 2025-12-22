"""
Tool Interaktif untuk Membuat Zona Parkir Ilegal
- Klik pada video untuk membuat titik zona
- Otomatis normalisasi koordinat
- Export zona untuk digunakan di detector
"""

import os
import cv2
import numpy as np
import json
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

class ZoneCreator:
    def __init__(self, cctv_name, video_path): # Menerima cctv_name dan video_path
        """Inisialisasi zone creator"""
        self.cctv_name = cctv_name # Simpan nama CCTV
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Baca frame pertama
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Tidak bisa membaca video!")
        
        self.original_frame = self.frame.copy()
        self.height, self.width = self.frame.shape[:2]
        
        # Data zona
        self.zones = []  # List of zones (setiap zona adalah list of points)
        self.current_zone = []  # Points untuk zona yang sedang dibuat
        self.zone_colors = [
            (0, 0, 255),    # Merah
            (0, 255, 0),    # Hijau
            (255, 0, 0),    # Biru
            (0, 255, 255),  # Kuning
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
        self.window_name = 'Tools Zoning Parkir Ilegal'
        
        print("\n" + "="*60)
        print("TOOL PEMBUAT ZONA PARKIR ILEGAL")
        print("="*60)
        print("\nResolusi Video:", f"{self.width}x{self.height}")
        print("\nPETUNJUK:")
        print("  - Klik KIRI: Tambah titik zona")
        print("  - Klik KANAN: Hapus titik terakhir")
        print("  - Tekan 'C': Selesai zona (minimal 3 titik)")
        print("  - Tekan 'Z': Hapus zona terakhir")
        print("  - Tekan 'R': Reset semua zona")
        print("  - Tekan 'S': Simpan zona ke file")
        print("  - Tekan 'Q': Keluar")
        print("="*60 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback untuk mouse event"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Tambah titik
            self.current_zone.append((x, y))
            print(f"Titik ditambahkan: ({x}, {y}) | "
                  f"Normalized: ({x/self.width:.3f}, {y/self.height:.3f})")
            self.draw_frame()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Hapus titik terakhir
            if self.current_zone:
                removed = self.current_zone.pop()
                print(f"Titik dihapus: {removed}")
                self.draw_frame()
    
    def normalize_point(self, point):
        """Normalisasi koordinat pixel ke 0-1"""
        return (point[0] / self.width, point[1] / self.height)
    
    def denormalize_point(self, point_norm):
        """Denormalisasi koordinat 0-1 ke pixel"""
        return (int(point_norm[0] * self.width), 
                int(point_norm[1] * self.height))
    
    def draw_frame(self):
        """Gambar frame dengan zona"""
        self.frame = self.original_frame.copy()
        
        # Gambar zona yang sudah selesai
        for i, zone in enumerate(self.zones):
            color = self.zone_colors[i % len(self.zone_colors)]
            
            # Gambar poligon dengan transparansi
            overlay = self.frame.copy()
            zone_array = np.array(zone, dtype=np.int32)
            cv2.fillPoly(overlay, [zone_array], color)
            cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
            
            # Gambar outline
            cv2.polylines(self.frame, [zone_array], True, color, 2)
            
            # Gambar titik-titik
            for point in zone:
                cv2.circle(self.frame, point, 5, color, -1)
            
            # Label zona
            centroid = np.mean(zone_array, axis=0).astype(int)
            cv2.putText(self.frame, f"ZONA {i+1}", tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Gambar zona yang sedang dibuat
        if self.current_zone:
            color = self.zone_colors[len(self.zones) % len(self.zone_colors)]
            
            # Gambar garis antar titik
            for i in range(len(self.current_zone) - 1):
                cv2.line(self.frame, self.current_zone[i], 
                        self.current_zone[i+1], color, 2)
            
            # Garis dari titik terakhir ke titik pertama (preview)
            if len(self.current_zone) >= 3:
                cv2.line(self.frame, self.current_zone[-1], 
                        self.current_zone[0], color, 1, cv2.LINE_AA)
            
            # Gambar titik-titik
            for point in self.current_zone:
                cv2.circle(self.frame, point, 5, color, -1)
                cv2.circle(self.frame, point, 7, (255, 255, 255), 1)
        
        # Info di layar
        info_y = 30
        cv2.putText(self.frame, f"Zona Selesai: {len(self.zones)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.current_zone:
            cv2.putText(self.frame, f"Titik Zona Saat Ini: {len(self.current_zone)}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, self.frame)
    
    def complete_zone(self):
        """Selesaikan zona yang sedang dibuat"""
        if len(self.current_zone) < 3:
            print("ERROR: Minimal 3 titik untuk membuat zona!")
            return False
        
        self.zones.append(self.current_zone.copy())
        print(f"\nZona {len(self.zones)} selesai dibuat dengan {len(self.current_zone)} titik")
        
        # Cetak koordinat normalized
        print("Koordinat Normalized:")
        for i, point in enumerate(self.current_zone):
            norm = self.normalize_point(point)
            print(f"  {i+1}. ({norm[0]:.4f}, {norm[1]:.4f})")
        
        self.current_zone = []
        self.draw_frame()
        return True
    
    def remove_last_zone(self):
        """Hapus zona terakhir"""
        if self.zones:
            removed = self.zones.pop()
            print(f"Zona {len(self.zones)+1} dihapus")
            self.draw_frame()
        else:
            print("Tidak ada zona untuk dihapus")
    
    def reset_all(self):
        """Reset semua zona"""
        self.zones = []
        self.current_zone = []
        print("Semua zona direset")
        self.draw_frame()
    
    def save_zones(self): # filename akan dibuat secara dinamis
        """Simpan zona ke file JSON"""
        if not self.zones:
            print("Tidak ada zona untuk disimpan!")
            return
        
        filename = f'hasil_zoning/parking_zones_{self.cctv_name}.json'
        
        # Pastikan direktori output ada
        output_dir = os.path.dirname(filename)
        if output_dir: # Hanya buat direktori jika path_file menyertakan nama direktori
            os.makedirs(output_dir, exist_ok=True)
        
        # Konversi ke normalized coordinates
        normalized_zones = []
        for zone in self.zones:
            norm_zone = [self.normalize_point(point) for point in zone]
            normalized_zones.append(norm_zone)
        
        # Simpan ke JSON
        data = {
            'video_info': {
                'width': self.width,
                'height': self.height,
                'path': self.video_path,
                'cctv_name': self.cctv_name
            },
            'zones': normalized_zones
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Zona disimpan ke: {filename}")
        print(f"Total zona: {len(normalized_zones)}")
        print(f"{'='*60}\n")
        
        # Cetak kode Python untuk langsung digunakan
        print("Koordinat Zoning Parkir Ilegal:")
        print("-" * 60)
        for i, zone in enumerate(normalized_zones):
            print(f"zone{i+1} = [")
            for point in zone:
                print(f"    ({point[0]:.4f}, {point[1]:.4f}),")
            print(f"]")
            print(f"detector.add_zone(zone{i+1})")
            print()
        print("-" * 60)
    
    def load_zones(self): # filename akan dibuat secara dinamis
        """Load zona dari file JSON"""
        filename = f'hasil_zoning/parking_zones_{self.cctv_name}.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Konversi normalized ke pixel coordinates
            self.zones = []
            for norm_zone in data['zones']:
                pixel_zone = [self.denormalize_point(point) for point in norm_zone]
                self.zones.append(pixel_zone)
            
            print(f"Berhasil load {len(self.zones)} zona dari {filename}")
            self.draw_frame()
        except FileNotFoundError:
            print(f"File {filename} tidak ditemukan. Memulai dengan zona kosong.")
        except Exception as e:
            print(f"Error loading zones: {e}")
    
    
    def run(self):
        """Jalankan Tools Zoning"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, 
                            self.mouse_callback)
        
        self.draw_frame()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Keluar
                break
            
            elif key == ord('c'):
                # Complete zona
                self.complete_zone()
            
            elif key == ord('z'):
                # Hapus zona terakhir
                self.remove_last_zone()
            
            elif key == ord('r'):
                # Reset semua
                self.reset_all()
            
            elif key == ord('s'):
                # Simpan zona
                self.save_zones()
            
            elif key == ord('l'):
                # Load zona
                self.load_zones()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        return self.zones


# ============================================
# CONTOH PENGGUNAAN
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Interaktif untuk Membuat Zona Parkir Ilegal")
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
    
    try:
        # Buat zona creator
        creator = ZoneCreator(args.cctv_name, video_path)
        
        # Jalankan tool
        zones = creator.run()
        
        print("\n" + "="*60)
        print("SELESAI!")
        print(f"Total zona dibuat: {len(zones)}")
        print("="*60)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")