from data_loader import load_pet_data
from utils import plot_samples
from image_preprocessing import create_dataset, load_and_preprocess_image
import tensorflow as tf
from model import create_model, train_model
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Load dataset
df = load_pet_data("PetImages")

# Convert DataFrame columns to lists
paths = df['images'].tolist()
labels = df['label'].tolist()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42
)

train_dataset = create_dataset(train_paths, train_labels, augment=True, shuffle=True)
val_dataset = create_dataset(val_paths, val_labels, augment=False, shuffle=False)

# Create model
model = create_model()
history = train_model(model, train_dataset, val_dataset, epochs=30)
