import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

model = load_model("model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Image not found ❌"

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    prediction = model.predict(img)[0][0]

    confidence = prediction * 100

    if prediction > 0.5:
        return f"FAKE ❌ ({confidence:.2f}%)"
    else:
        return f"REAL ✅ ({100-confidence:.2f}%)"

if len(sys.argv) < 2:
    print("Usage: python main.py image.jpg")
else:
    print(predict_image(sys.argv[1]))