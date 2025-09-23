import os
import pandas as pd
from PIL import Image

def load_pet_data(root="PetImages"):
    """
    Loads PetImages dataset into a pandas DataFrame with columns:
    - 'images': file paths
    - 'label': 0 for Cat, 1 for Dog

    Automatically handles palette-based images (mode 'P') by converting to RGB.
    Skips corrupted or unreadable images.
    """
    input_path, label = [], []

    for petclass in os.listdir(root):
        cls_folder = os.path.join(root, petclass)
        if not os.path.isdir(cls_folder):
            continue

        for file in os.listdir(cls_folder):
            file_path = os.path.join(cls_folder, file)

            # Skip non-files (like folders)
            if not os.path.isfile(file_path):
                continue

            # Attempt to open and validate image
            try:
                img = Image.open(file_path)

                # Convert palette-based or other unsupported modes to RGB
                if img.mode not in ("L", "RGB", "RGBA"):
                    img = img.convert("RGB")

                img.verify()  # checks for corruption
            except Exception:
                print(f"Skipping {file_path} (corrupt or unreadable)")
                continue

            # Valid image, add to dataset
            input_path.append(file_path)
            label.append(0 if petclass.lower() == "cat" else 1)

    # Create DataFrame and shuffle
    df = pd.DataFrame({"images": input_path, "label": label})
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Loaded {len(df)} valid images.")
    return df
