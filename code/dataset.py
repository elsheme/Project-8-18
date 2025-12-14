import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

BASE_PATH = os.path.join(os.path.dirname(__file__), "../Datasets")
IMG_SIZE = (128, 128)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32)

def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(BASE_PATH))
    class_map = {c: i for i, c in enumerate(classes)}

    for cls in classes:
        folder = os.path.join(BASE_PATH, cls)
        for img_name in os.listdir(folder):
            img = load_image(os.path.join(folder, img_name))
            if img is not None:
                X.append(img)
                y.append(class_map[cls])

    X = np.array(X)
    y = to_categorical(y, len(classes))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    print(f"Total images: {len(X)}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def training_generator(X, y, batch_size=32):
    while True:
        idx = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_x = X[idx[i:i + batch_size]]
            batch_y = y[idx[i:i + batch_size]]
            yield batch_x, batch_y