from ultralytics import YOLO

# Load model yang sudah dilatih dengan 80 kelas COCO
model = YOLO('yolov8x.pt')  

# Filter hanya kendaraan saat deteksi
results = model('/home/ozzaann/learn-yolo/test-sample/bogor_vehicle.png', classes=[2], conf=0.5, save=True)