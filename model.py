import tensorflow as tf
import os
from utils import IMG_SIZE, WEIGHTS_FILE


def create_model():
    INPUT_SHAPE = IMG_SIZE + (3,)  # (x, y, 3)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),
        
        # Convolution + Pooling
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        # Flatten + Dense layers
        tf.keras.layers.Flatten(),
        #tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        
        # Output layer for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    return model


# ---------------------------
# TRAIN MODEL FUNCTION
# ---------------------------
def train_model(model, train_dataset, val_dataset, epochs, weights_file=WEIGHTS_FILE):
    """
    Trains the model and saves the weights for later inference.

    Args:
        model: compiled tf.keras.Model
        train_dataset: tf.data.Dataset for training
        val_dataset: tf.data.Dataset for validation
        epochs: number of epochs to train
        weights_file: path to save the model weights

    Returns:
        history: training history object
    """
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1 # print progress bar
    )

    # Save weights
    model.save_weights(weights_file)
    print(f"Weights saved to {weights_file}")

    return history


def predict_image(model, image_tensor):
    """
    Predicts whether an image is a cat or dog.
    
    Args:
        model: Trained TensorFlow model
        image_tensor: Preprocessed image tensor of shape (1, 128, 128, 3)
    
    Returns:
        String: "Cat" or "Dog"
    """
    prediction = model.predict(image_tensor, verbose=0)
    # prediction is a value between 0 and 1
    # Typically: < 0.5 = Cat, >= 0.5 = Dog
    if prediction[0][0] < 0.5:
        return "Cat"
    else:
        return "Dog"
    
def predict_with_confidence(model, image_tensor):
    """
    Predicts whether an image is a cat or dog, along with confidence score.
    
    Args:
        model: Trained TensorFlow model
        image_tensor: Preprocessed image tensor of shape (1, 128, 128, 3)
    
    Returns:
        tuple: (label, confidence)
            label: "Cat" or "Dog"
            confidence: float between 0 and 1
    """
    prediction = model.predict(image_tensor, verbose=0)[0][0]

    if prediction < 0.5:
        label = "Cat"
        confidence = 1 - prediction  # probability of being Cat
    else:
        label = "Dog"
        confidence = prediction      # probability of being Dog
    
    return label, float(confidence)
