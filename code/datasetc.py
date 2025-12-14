import cv2
import random
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

EGYPTIAN_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./Datasets/")
INDIAN_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Datasets/")

IMG_SIZE = (128, 128)
MAX_SAMPLES_PER_CLASS = 100


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img / 255.0


def load_data_partially(base_path, class_to_idx, max_samples, is_nested=False, subset_folder=None):
    X, Y = [], []

    if is_nested and subset_folder:
        data_path = os.path.join(base_path, subset_folder)
    else:
        data_path = base_path

    if not os.path.exists(data_path):
        print(f"Error: Path not found: {data_path}")
        return np.array(X), np.array(Y)

    classes = sorted(os.listdir(data_path))

    for cls_name in classes:
        class_path = os.path.join(data_path, cls_name)
        if not os.path.isdir(class_path): continue
        if cls_name not in class_to_idx: continue

        images = os.listdir(class_path)
        selected_images = random.sample(images, min(len(images), max_samples))
        cls_idx = class_to_idx[cls_name]

        for img_name in selected_images:
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                X.append(img)
                Y.append(cls_idx)

    return np.array(X, dtype=np.float32), np.array(Y)


def load_combined_dataset():
    indian_classes = ['fifty_new', 'fifty_old', 'five_hundred', 'hundred_new', 'hundred_old',
                      'ten_new', 'ten_old', 'twenty_new', 'twenty_old', 'two_hundred', 'two_thousand']

    all_classes = []

    for cls in indian_classes:
        if cls not in all_classes:
            all_classes.append(cls)

    try:
        egyptian_classes = sorted(os.listdir(os.path.join(EGYPTIAN_BASE, 'train')))
        for cls in egyptian_classes:
            if cls not in all_classes:
                all_classes.append(cls)
    except FileNotFoundError:
        if len(all_classes) < 20:
            for i in range(9): all_classes.append(f'egyptian_coin_{i}')

    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    num_classes = len(class_to_idx)

    if num_classes != 20:
        print(f"FATAL ERROR: Found {num_classes} classes instead of 20. Check your paths.")
        return None, None, None, None, None, None, None

    X_indian, Y_indian = load_data_partially(INDIAN_BASE, class_to_idx, MAX_SAMPLES_PER_CLASS)

    X_i_train_val, X_i_test, y_i_train_val, y_i_test = train_test_split(
        X_indian, Y_indian, test_size=0.15, random_state=42, stratify=Y_indian)

    X_i_train, X_i_val, y_i_train, y_i_val = train_test_split(
        X_i_train_val, y_i_train_val, test_size=(0.15 / (1 - 0.15)), random_state=42, stratify=y_i_train_val)

    X_e_train, Y_e_train = load_data_partially(EGYPTIAN_BASE, class_to_idx, MAX_SAMPLES_PER_CLASS, is_nested=True,
                                               subset_folder='train')
    X_e_val, Y_e_val = load_data_partially(EGYPTIAN_BASE, class_to_idx, MAX_SAMPLES_PER_CLASS, is_nested=True,
                                           subset_folder='valid')
    X_e_test, Y_e_test = load_data_partially(EGYPTIAN_BASE, class_to_idx, MAX_SAMPLES_PER_CLASS, is_nested=True,
                                             subset_folder='test')

    X_train = np.concatenate([X_i_train, X_e_train])
    Y_train = np.concatenate([y_i_train, Y_e_train])

    X_val = np.concatenate([X_i_val, X_e_val])
    Y_val = np.concatenate([y_i_val, Y_e_val])

    X_test = np.concatenate([X_i_test, X_e_test])
    Y_test = np.concatenate([y_i_test, Y_e_test])

    X_train, Y_train = X_train, to_categorical(Y_train, num_classes=num_classes)
    X_val, Y_val = X_val, to_categorical(Y_val, num_classes=num_classes)
    X_test, Y_test = X_test, to_categorical(Y_test, num_classes=num_classes)

    p = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[p], Y_train[p]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, all_classes


def training_generator(X_data, Y_data, batch_size):
    data_len = len(X_data)
    while True:
        indices = np.random.permutation(data_len)

        for i in range(0, data_len, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X_data[batch_indices], Y_data[batch_indices]


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_combined_dataset()

    assert len(class_names) == 20

    print(f"Total Unified Training Samples: {len(X_train)}")
    print("Data loading complete. Ready for unified model training.")