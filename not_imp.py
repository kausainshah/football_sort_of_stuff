import cv2
import numpy as np
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    # converting to numpy type array
    images = np.array(images)
    return images


im = load_images_from_folder("del/")
