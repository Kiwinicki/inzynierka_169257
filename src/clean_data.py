from .dataset import FERDataset
import torch
import torchvision
import cv2
import numpy as np

ds = FERDataset("./data")

img, label = ds[0]
print(torchvision.transforms.functional.pil_to_tensor(img).shape, label)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def filter_by_face_detection(img):
    img = np.array(img)
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    faces = face_cascade.detectMultiScale(img_uint8, 1.1, 4)
    return len(faces) > 0


print(filter_by_face_detection(img))
