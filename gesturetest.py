from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import cv2

model = load_model("vgg-rps-1.h5")

path = "rps-final-dataset/rps-results"
i = 0

for img in os.listdir(path):
    img = image.load_img(path + "/" + img, target_size=(224, 224))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=1)
    if pred[0][0] > 0.5:
        category = "Paper"
    elif pred[0][1] > 0.5:
        category = "Rock"
    elif pred[0][2] > 0.5:
        category = "Scissors"

    print(category)