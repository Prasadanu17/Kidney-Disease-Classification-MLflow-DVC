import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load trained model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # PREPROCESSING FIX: apply the same VGG16 preprocessing used at
        # training time (preprocess_input) instead of raw pixel values.
        # Previously this was missing entirely, causing a train/inference mismatch.
        test_image = preprocess_input(test_image)

        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = "Tumor"
            return [{"image": prediction}]
        else:
            prediction = "No Tumor"
            return [{"image": prediction}]