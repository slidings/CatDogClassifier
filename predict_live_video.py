import cv2
import tensorflow as tf
from model import create_model, predict_image, predict_with_confidence
from utils import IMG_SIZE, WEIGHTS_FILE

# ---------------------------
# Load model architecture and weights
# ---------------------------
model = create_model()
model.load_weights(WEIGHTS_FILE)
print(f"Weights loaded from {WEIGHTS_FILE}")

# ---------------------------
# Helper function to preprocess a frame
# ---------------------------
def preprocess_image_from_array(img_array):
    """
    Converts BGR image from OpenCV to RGB, resizes, normalizes, and adds batch dimension.
    """
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

# ---------------------------
# Capture from camera and classify continuously
# ---------------------------
def live_camera_classification():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess frame
        img = preprocess_image_from_array(frame)

        # Predict with confidence
        label, confidence = predict_with_confidence(model, img)

        # Display label + confidence on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Camera - Cat/Dog Classifier", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    live_camera_classification()
