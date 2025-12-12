import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

##os.system("clear")


BASE_PATH = "../Datasets/"
IMG_SIZE = (128, 128)
AUGMENT = True



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
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    if np.random.rand() < 0.4:
        factor = 0.5 + np.random.rand() * 1.0
        img = np.clip(img * factor, 0, 1)

    return img


def show_example_images():
    if not os.path.exists(BASE_PATH):
        print(f"The directory {BASE_PATH} doesn't exist.")

    classes = sorted(os.listdir(BASE_PATH))
    NUM_IMAGES = 3

    print("**" * 20)
    print(" " * 9, f"Dataset path: {BASE_PATH}")
    print(" " * 9, f"Total classes = {len(classes)}")
    print("**" * 20)


    fig, ax = plt.subplots(nrows=len(classes), ncols=NUM_IMAGES, figsize=(10, 30))

    for p, c in enumerate(classes):
        total_images_class = len(os.listdir(BASE_PATH +c))
        print(f"* {c} => {total_images_class} images")

        imgs = os.listdir(os.path.join(BASE_PATH, c))
        selected = random.choices(imgs, k=NUM_IMAGES)

        for i, img_name in enumerate(selected):
            img_path = os.path.join(BASE_PATH, c, img_name)
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ax[p, i].imshow(img_rgb)
            ax[p, i].set_title(f"{c}\n{img_rgb.shape}")
            ax[p, i].axis("off")

    # plt.tight_layout()
    # fig.show()


def load_dataset():
    X, Y = [], []

    classes = sorted(os.listdir(BASE_PATH))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    print("\nLoading images...\n")

    for cls in classes:
        class_path = os.path.join(BASE_PATH, cls)
        images = os.listdir(class_path)

        for img_name in images:
            img_path = os.path.join(class_path, img_name)

            img = preprocess_image(img_path)

            X.append(img)
            Y.append(class_to_idx[cls])

    X = np.array(X)
    Y = to_categorical(Y, num_classes=len(classes))

    print("Dataset loaded successfully!")
    print(f"Total images: {len(X)}")

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def training_generator(X, y, batch_size=32):
    while True:
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size].copy()
            batch_y = y[i:i + batch_size]

            for j in range(len(batch_X)):
                batch_X[j] = augment_image(batch_X[j])

            yield batch_X, batch_y



if __name__ == "__main__":

    show_example_images()

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()
