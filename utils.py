import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img

WEIGHTS_FILE = "cat_dog.weights.h5"
IMG_SIZE = (128, 128)

def plot_samples(df, label_value, title, n=25):
    """Plots n random images for a given label_value (0=cat, 1=dog)."""
    plt.figure(figsize=(25,25))
    temp = df[df['label'] == label_value]['images']
    start = random.randint(0, len(temp) - n)
    files = temp[start:start+n]

    for index, file in enumerate(files):
        plt.subplot(5, 5, index + 1)
        img = load_img(file)
        plt.imshow(np.array(img))
        plt.title(title)
        plt.axis('off')
    plt.show()
