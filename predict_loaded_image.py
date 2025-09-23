import tensorflow as tf
from model import create_model, predict_image
from utils import IMG_SIZE, WEIGHTS_FILE

# ---------------------------
# Load model architecture and weights
# ---------------------------
model = create_model()          # same architecture as training
model.load_weights(WEIGHTS_FILE)
print(f"Weights loaded from {WEIGHTS_FILE}")

# ---------------------------
# Helper function to preprocess a single image
# ---------------------------
def preprocess_single_image(img_path):
    """
    Reads and preprocesses a single image for prediction.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)  # shape: (1, 128, 128, 3)
    return img

# ---------------------------
# Run inference
# ---------------------------
def classify_image(img_path):
    img = preprocess_single_image(img_path)
    result = predict_image(model, img)
    print(f"The image '{img_path}' is predicted as: {result}")
    return result

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    test_image_path = "cat1.jpg"
    classify_image(test_image_path)
