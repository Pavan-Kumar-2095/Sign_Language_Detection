import cv2
import os
import time

# Create dataset directory if it doesn't exist
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def capture_images(label, num_images=101, delay=1):
    """Capture 'num_images' images and save them in a folder named after 'label'."""
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)  # Open webcam
    count = 1
    
    print(f"Capturing images for '{label}'... Look at the camera.")
    time.sleep(2)  # Give user time to adjust

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing webcam.")
            break

        img_path = os.path.join(label_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)  # Save image
        count += 1

        cv2.imshow("Capturing...", frame)
        time.sleep(delay)  # Wait before next capture

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {num_images} images in '{label_dir}'.")

# Get user input for the sign label
label = input("Enter the label (e.g., A, B, 1, 2): ").strip()
capture_images(label)