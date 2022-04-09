import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential

from PIL import Image
import numpy as np

model_path = "../common/cnn_finalized_model_BEST.sav"
image_path = "../build-gui-Desktop-Debug/selected_image.png"
model = keras.models.load_model(model_path)

def classify():
    im = Image.open(image_path)
    pixel_colors = np.zeros((1, 50, 50, 3), dtype = float)
    pixel_colors[0, :] = np.asarray(im, dtype = float) / 255.0
    im.close()
    prediction = model.predict(pixel_colors)[0][0]
    return prediction



