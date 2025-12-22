import cv2
from ultralytics import YOLO

# Load model yang sudah dilatih
model = YOLO('yolov8x.pt')

# Path ke video input
cap = cv2.VideoCapture('test-sample/test.mp4')

# Loop untuk memproses setiap frame dari video
while cap.isOpened():
    # Baca frame dari video
    success, frame = cap.read()

    if success:
        # Lakukan deteksi objek pada frame
        results = model(frame, classes=[1], conf=0.2)

        # Anotasi frame dengan hasil deteksi
        annotated_frame = results[0].plot()

        # Tampilkan frame yang sudah di-anotasi
        cv2.imshow("Vehicle Detection YOLOv8", annotated_frame)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Keluar dari loop jika sudah di akhir video
        break

# Lepaskan video capture dan tutup semua window
cap.release()
cv2.destroyAllWindows()