import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

class ParkingDetector:
    def __init__(self, config_path="config/parking_ilegal/config.yaml", camera_id=None, zones_path=None):
        # Load configs (simplified for this script, assumes a dict or parses yaml)
        # In a real setup, config would be passed down from ai_manager
        self.camera_id = camera_id
        
        # Default config fallback
        self.model_path = "models/parking_ilegal/yolov8s.pt"
        self.conf_threshold = 0.2
        self.violation_threshold = 10
        self.vehicle_classes = ['car']
        self.vehicle_class_indices = [2]
        self.tracker_config = "bytetrack.yaml"

        self.model = YOLO(self.model_path)
        self.illegal_zones = []
        self.vehicles_in_zone = {} # {track_id: {'entry_time': float, 'class': str, 'zone_id': int}}
        self.logged_violations = set()
        self.total_violations = 0

        # Load zones
        if zones_path:
            self.load_zones(zones_path)

    def load_zones(self, zones_file):
        try:
            with open(zones_file, 'r') as f:
                data = json.load(f)
            zones_from_json = data.get('zones', [])
            for zone in zones_from_json:
                zone_as_tuples = [tuple(point) for point in zone]
                self.illegal_zones.append(np.array(zone_as_tuples, dtype=np.float32))
            print(f"[{self.camera_id}] Loaded {len(self.illegal_zones)} zones.")
        except FileNotFoundError:
            print(f"[{self.camera_id}] Error: Zone file '{zones_file}' not found.")
        except Exception as e:
            print(f"[{self.camera_id}] Error loading zones: {e}")

    def point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def get_box_bottom_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y2)

    def denormalize_zone(self, zone_norm, frame_shape):
        height, width, _ = frame_shape
        return np.array([ (int(pt[0] * width), int(pt[1] * height)) for pt in zone_norm ], dtype=np.int32)

    def draw_zones(self, frame):
        overlay = frame.copy()
        for i, zone_norm in enumerate(self.illegal_zones):
            zone_pixels = self.denormalize_zone(zone_norm, frame.shape)
            cv2.fillPoly(overlay, [zone_pixels], (0, 0, 255))
            cv2.polylines(frame, [zone_pixels], True, (0, 0, 255), 2)
            centroid = np.mean(zone_pixels, axis=0).astype(int)
            cv2.putText(frame, f"ZONA ILEGAL {i+1}", (centroid[0] - 50, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        return frame

    def format_duration(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def process_frame(self, frame):
        """Processes a single frame, updates tracking and detection state, returns annotated frame."""
        current_time_sec = time.time()
        
        results = self.model.track(
            frame, 
            persist=True, 
            classes=self.vehicle_class_indices, 
            conf=self.conf_threshold,
            tracker=self.tracker_config,
            verbose=False
        )
        
        frame_height, frame_width, _ = frame.shape
        current_frame_track_ids = set()

        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, track_ids, classes):
                current_frame_track_ids.add(track_id)
                class_name = self.model.names[cls]
                x1, y1, x2, y2 = box

                center_pixel = self.get_box_bottom_center(box)
                center_norm = (center_pixel[0] / frame_width, center_pixel[1] / frame_height)
                
                in_illegal_zone = False
                zone_idx = -1
                for i, zone_norm in enumerate(self.illegal_zones):
                    if self.point_in_polygon(center_norm, zone_norm):
                        in_illegal_zone = True
                        zone_idx = i
                        break
                
                if in_illegal_zone:
                    if track_id not in self.vehicles_in_zone:
                        self.vehicles_in_zone[track_id] = {
                            'entry_time': current_time_sec,
                            'class': class_name,
                            'zone_id': zone_idx
                        }
                    
                    duration = current_time_sec - self.vehicles_in_zone[track_id]['entry_time']
                    is_violation = duration >= self.violation_threshold
                    color = (0, 0, 255) if is_violation else (0, 165, 255)
                    
                    if is_violation and track_id not in self.logged_violations:
                        # violation_logger.log_violation(...) -> handled via callbacks or event queue in clean arch
                        # For now we just print
                        print(f"[{self.camera_id}] VIOLATION: ID {track_id} exceeded threshold.")
                        self.logged_violations.add(track_id)
                        self.total_violations += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} | ID: {track_id} | LIMIT: {self.format_duration(duration)}" if is_violation else f"{class_name} | ID: {track_id} | {self.format_duration(duration)}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                else:
                    if track_id in self.vehicles_in_zone:
                        del self.vehicles_in_zone[track_id]
                        if track_id in self.logged_violations:
                            self.logged_violations.remove(track_id)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} | ID: {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        lost_track_ids = set(self.vehicles_in_zone.keys()) - current_frame_track_ids
        for track_id in lost_track_ids:
            del self.vehicles_in_zone[track_id]
            if track_id in self.logged_violations:
                self.logged_violations.remove(track_id)

        frame = self.draw_zones(frame)
        cv2.putText(frame, f"Kendaraan di Zona Ilegal: {len(self.vehicles_in_zone)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total Pelanggar: {self.total_violations}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return frame
