import os
import glob
import cv2
from ultralytics import YOLO

# 1. Locate best.pt
weights_path = "best.pt"

if not os.path.exists(weights_path):
    existing_weights = glob.glob("**/best.pt", recursive=True)
    if existing_weights:
        weights_path = existing_weights[0]
        print(f"Using existing weights found at: {weights_path}")
    else:
        raise FileNotFoundError("Error: 'best.pt' not found in current directory or subfolders.")

# 2. Load YOLO Model
print(f"Loading model from {weights_path}...")
model = YOLO(weights_path)
print(f"Model classes: {model.names}")

# 3. Initialize Webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

# Optional: Set camera width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("\nStarting live webcam stream... Press 'q' while focused on the window to exit.\n")

# 4. Stream and Inference Loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Run YOLO inference on the current frame
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # Render bounding boxes onto the frame
    annotated_frame = results[0].plot()

    # Display the frame in a local window
    cv2.imshow("YOLO Local Webcam Test", annotated_frame)

    # Press 'q' on your keyboard to terminate the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Release Resources
cap.release()
cv2.destroyAllWindows()