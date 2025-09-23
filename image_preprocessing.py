import tensorflow as tf
from PIL import Image
import numpy as np
import math
from utils import IMG_SIZE

def load_and_preprocess_image(path, label):
    """
    Robust image loading function that handles all formats and shape issues
    """
    # Use tf.py_function to wrap our custom loading logic
    def load_image_py_func(path_tensor, label_tensor):
        # Convert tensors to numpy values
        path_str = path_tensor.numpy().decode('utf-8')
        label_val = label_tensor.numpy()
        
        try:
            # Use PIL for maximum compatibility
            with Image.open(path_str) as img:
                # Convert to RGB (handles all formats)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize and normalize
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                return img_array, float(label_val)
                
        except Exception as e:
            # Return a proper black image on any error
            print(f"Error loading {path_str}: {e}")
            return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32), float(label_val)
    
    # Use tf.py_function instead of tf.numpy_function
    image, label = tf.py_function(
        load_image_py_func, 
        [path, label], 
        [tf.float32, tf.float32]
    )
    
    # Explicitly set shapes for both image and label
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    label.set_shape(())
    
    return image, label


# Augment to improve generalisability of model
def augment_image(image, label):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # Random rotation by 0, 90, 180, 270 degrees
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # Random zoom by central crop
    zoom = tf.random.uniform([], 0.8, 1.0)
    crop_size = tf.cast(zoom * IMG_SIZE[0], tf.int32)
    image = tf.image.central_crop(image, central_fraction=zoom)
    image = tf.image.resize(image, IMG_SIZE)

    return image, label

def create_dataset(paths, labels, batch_size=32, augment=False, shuffle=True):
    """
    Converts a dataframe of image paths and labels into a TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda p, l: load_and_preprocess_image(p, l),
                          num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Outdated way of preprocessing
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define preprocessing + augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,        # normalize pixels
    rotation_range=40,     # random rotation
    shear_range=0.2,       # shear
    zoom_range=0.2,        # zoom
    horizontal_flip=True,  # flip horizontally
    fill_mode='nearest'    # fill empty pixels
)

# Use your DataFrame df
train_iterator = train_gen.flow_from_dataframe(
    df,
    x_col='images',
    y_col='label',
    target_size=(128,128), # resize all images
    batch_size=32,
    class_mode='binary'
)
'''