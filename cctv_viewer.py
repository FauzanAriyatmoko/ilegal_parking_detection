import cv2
import json

def main():
    # Load CCTV sources from JSON file
    try:
        with open('cctv_sources.json', 'r') as f:
            cctv_sources = json.load(f)
    except FileNotFoundError:
        print("Error: cctv_sources.json tidak ditemukan.")
        return
    except json.JSONDecodeError:
        print("Error: Format JSON di cctv_sources.json tidak valid.")
        return

    source_names = list(cctv_sources.keys())
    source_urls = list(cctv_sources.values())
    current_source_index = 0

    # Create a resizable window
    window_name = 'CCTV Bogor Steaming Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = None  # Initialize cap to None

    # Function to switch CCTV stream
    def switch_stream(index):
        nonlocal cap
        if cap is not None and cap.isOpened():
            cap.release()
        
        print(f"\nSwitching to: {source_names[index]} ({source_urls[index]})")
        cap = cv2.VideoCapture(source_urls[index])
        
        if not cap.isOpened():
            print(f"Error: Could not open stream for {source_names[index]}")
        return cap

    # Initialize video capture with the first source
    cap = switch_stream(current_source_index)

    while True:
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Display the current CCTV name on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{source_names[current_source_index]}"
                cv2.putText(frame, text, (10, 100), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
            else:
                # If the stream fails, show a black screen with an error message
                error_frame = cv2.UMat(500, 900, cv2.CV_8UC3)
                error_frame.setTo((0, 0, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"Gagal memuat stream: {source_names[current_source_index]}"
                cv2.putText(error_frame, text, (10, 100), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow(window_name, error_frame)
        else:
            # Handle case where cap is not initialized at all
            error_frame = cv2.UMat(500, 900, cv2.CV_8UC3)
            error_frame.setTo((0, 0, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Tidak ada stream yang aktif. Tekan 's' untuk mencari."
            cv2.putText(error_frame, text, (10, 100), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow(window_name, error_frame)


        key = cv2.waitKey(25) & 0xFF

        # Quit program
        if key == ord('q'):
            print("Keluar dari program.")
            break
        # Next CCTV
        elif key == ord('d'):
            current_source_index = (current_source_index + 1) % len(source_urls)
            cap = switch_stream(current_source_index)
        # Previous CCTV
        elif key == ord('a'):
            current_source_index = (current_source_index - 1 + len(source_urls)) % len(source_urls)
            cap = switch_stream(current_source_index)
        # Search for a CCTV location
        elif key == ord('s'):
            print("\n--- Mode Pencarian ---")
            print("Lokasi yang tersedia:")
            for name in source_names:
                print(f"- {name}")
            
            search_term = input("Search nama lokasi CCTV Bogor: ").strip().lower()
            
            if not search_term:
                print("Pencarian dibatalkan.")
                continue

            found_index = -1
            for i, name in enumerate(source_names):
                if search_term in name.lower():
                    found_index = i
                    break
            
            if found_index != -1:
                current_source_index = found_index
                cap = switch_stream(current_source_index)
            else:
                print(f"Lokasi '{search_term}' tidak ditemukan.")

    # Release resources
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
