import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = None

    def _load_model(self):
        if self.model is None:
            model_path = os.path.join("artifacts", "training", "model.h5")
            # compile=False avoids Keras deserialization errors (e.g., reduction='auto')
            self.model = load_model(model_path, compile=False)
        return self.model

    def predict(self):
        model = self._load_model()

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # PREPROCESSING: apply VGG16 preprocess_input for fine-tuned model
        test_image = preprocess_input(test_image)

        preds = model.predict(test_image, verbose=0)
        result = np.argmax(preds, axis=1)
        conf = float(np.max(preds))
        print(f"Prediction result: {result}, probabilities: {preds}")

        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction, "confidence": round(conf * 100, 2)}]