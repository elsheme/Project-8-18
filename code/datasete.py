import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# =========================================================================
# 1. CONFIGURATION AND DOWNLOAD SETUP
# =========================================================================
DATASET_NAME = 'belalsafy/egyptian-new-currency-2023'
DOWNLOAD_PATH = './Datasets'
BASE_PATH = DOWNLOAD_PATH
IMG_SIZE = (128, 128)
AUGMENT = True


def download_dataset_if_not_exists():
    if os.path.isdir(DOWNLOAD_PATH) and os.listdir(DOWNLOAD_PATH):
        print("Dataset already exists in the path:", DOWNLOAD_PATH)
        return

    print(f"Dataset not found. Connecting to Kaggle API to download: {DATASET_NAME}...")

    try:
        api = KaggleApi()
        api.authenticate()

        os.makedirs(DOWNLOAD_PATH, exist_ok=True)

        api.dataset_download_files(
            dataset=DATASET_NAME,
            path=DOWNLOAD_PATH,
            unzip=True
        )
        print(f"Dataset downloaded successfully to folder: {DOWNLOAD_PATH}")

    except Exception as e:
        print(f"An error occurred during Kaggle API download: {e}")
        print(
            "Please ensure Kaggle API authentication is correctly set up for all team members (kaggle.json in ~/.kaggle).")


# =========================================================================
# 2. PREPROCESSING AND AUGMENTATION FUNCTIONS
# =========================================================================

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img


def augment_image(img):
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    if np.random.rand() < 0.3:
        angle = np.random.randint(-20, 20)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    if np.random.rand() < 0.4:
        factor = 0.5 + np.random.rand() * 1.0
        img = np.clip(img * factor, 0, 1)

    return img


# =========================================================================
# 3. DATA LOADING AND SPLITTING
# =========================================================================

def show_example_images():
    if not os.path.exists(BASE_PATH):
        print(f"The directory {BASE_PATH} doesn't exist. Cannot show examples.")
        return

    data_subfolders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]

    if not data_subfolders:
        print(f"No class folders found in {BASE_PATH}.")
        return

    actual_data_path = os.path.join(BASE_PATH, data_subfolders[0])

    classes = sorted(os.listdir(actual_data_path))
    NUM_IMAGES = 3

    print("\n" + "**" * 20)
    print(" " * 9, f"Dataset path: {actual_data_path}")
    print(" " * 9, f"Total classes = {len(classes)}")
    print("**" * 20)

    fig, ax = plt.subplots(nrows=len(classes), ncols=NUM_IMAGES, figsize=(10, 30))

    for p, c in enumerate(classes):
        class_path = os.path.join(actual_data_path, c)
        total_images_class = len(os.listdir(class_path))
        print(f"* {c} => {total_images_class} images")

        imgs = os.listdir(class_path)
        selected = random.choices(imgs, k=NUM_IMAGES)

        for i, img_name in enumerate(selected):
            img_path = os.path.join(class_path, img_name)
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ax[p, i].imshow(img_rgb)
            ax[p, i].set_title(f"{c}\n{img_rgb.shape}")
            ax[p, i].axis("off")

    plt.tight_layout()
    plt.show(block=False)


def load_dataset():
    X, Y = [], []

    data_subfolders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    if not data_subfolders:
        raise FileNotFoundError(f"Class folders not found inside {BASE_PATH}. Check dataset extraction.")

    actual_data_path = os.path.join(BASE_PATH, data_subfolders[0])

    classes = sorted(os.listdir(actual_data_path))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    print("\nLoading images...")

    for cls in classes:
        class_path = os.path.join(actual_data_path, cls)
        images = os.listdir(class_path)

        for img_name in images:
            img_path = os.path.join(class_path, img_name)

            img = preprocess_image(img_path)

            X.append(img)
            Y.append(class_to_idx[cls])

    X = np.array(X)
    Y = to_categorical(Y, num_classes=len(classes))

    print(f"Dataset loaded successfully! Total images: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================================
# 4. TRAINING GENERATOR
# =========================================================================

def training_generator(X, y, batch_size=32):
    while True:
        idx = np.random.permutation(len(X))
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i + batch_size].copy()
            batch_y = y_shuffled[i:i + batch_size]

            if AUGMENT:
                for j in range(len(batch_X)):
                    batch_X[j] = augment_image(batch_X[j])

            yield batch_X, batch_y


# =========================================================================
# MAIN EXECUTION BLOCK
# =========================================================================

if __name__ == "__main__":
    download_dataset_if_not_exists()

    show_example_images()

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()

