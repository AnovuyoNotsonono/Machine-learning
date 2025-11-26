#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("Tumor_detector.keras", compile=False)

def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(50, 50))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print("Predicted: Tumorous")
    else:
        print("Predicted: Negative")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(model, image_path)
    else:
        print("Please provide an image path as argument")

