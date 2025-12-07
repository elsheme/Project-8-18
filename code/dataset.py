import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import random


BASE_PATH = '../Datasets/'


if not os.path.exists(BASE_PATH):
    print(f"The directory {BASE_PATH} doesn't exist.")


print(f"The directory of data is : {BASE_PATH}")


classes = os.listdir(BASE_PATH)
classes = sorted(classes)
print("**" * 20)
print(" " * 9, f"Total classes = {len(classes)}")
print("**" * 20)
for c in classes:
    total_images_class = len(os.listdir(BASE_PATH +c))
    print(f"* {c} => {total_images_class} images")
NUM_IMAGES = 3
fig, ax = plt.subplots(nrows=len(classes), ncols=NUM_IMAGES, figsize=(10, 30))
p = 0
for c in classes:
    imgs = os.listdir(os.path.join(BASE_PATH, c) )
    imgs_selected = random.choices(imgs, k=NUM_IMAGES)
    for i, img in enumerate(imgs_selected):
        imgs = os.path.join(BASE_PATH, c, img)
        img_bgr = cv2.imread(str(imgs))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax[p, i].imshow(img_rgb)
        ax[p, i].set_title(f"Class: {c}\nShape: {img_rgb.shape}")
        ax[p, i].axis("off")

    p += 1

fig.tight_layout()
fig.show()
