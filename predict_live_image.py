import cv2
import tensorflow as tf
from model import create_model, predict_image
from utils import IMG_SIZE, WEIGHTS_FILE
# ---------------------------
# Load model architecture and weights
# ---------------------------
model = create_model()
model.load_weights(WEIGHTS_FILE)
print(f"Weights loaded from {WEIGHTS_FILE}")

# ---------------------------
# Helper function to preprocess a single image
# ---------------------------
def preprocess_image_from_array(img_array):
    """
    Takes a BGR image from OpenCV, converts to RGB, resizes and normalizes it.
    """
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)  # shape: (1, 128, 128, 3)
    return img

# ---------------------------
# Run inference
# ---------------------------
def classify_image_from_array(img_array):
    img = preprocess_image_from_array(img_array)
    result = predict_image(model, img)
    return result

# ---------------------------
# Capture from camera and classify
# ---------------------------
def capture_and_classify():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'c' to capture an image for classification or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Live Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Capture and classify
            result = classify_image_from_array(frame)
            print(f"Prediction: {result}")

            # Show captured image with prediction
            cv2.putText(frame, f"Prediction: {result}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Captured Image", frame)
            cv2.waitKey(3000)  # show for 3 seconds

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    capture_and_classify()
