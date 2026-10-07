import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
        Build training and validation data generators.

        PREPROCESSING FIX:
        - Previously used rescale=1./255 which produces pixel values in [0, 1].
        - VGG16 ImageNet weights expect inputs preprocessed with
          tf.keras.applications.vgg16.preprocess_input, which:
            1. Scales pixels to [0, 255] range (i.e. NO division by 255)
            2. Converts RGB -> BGR
            3. Subtracts ImageNet channel-wise mean: [103.939, 116.779, 123.68]
        - Using rescale=1./255 with ImageNet weights causes a severe input
          distribution mismatch, leading to poor validation accuracy (~51%).
        - This fix applies consistent VGG16 preprocessing to both train and
          validation generators (and must also be applied at inference time).
        """

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # ---- Validation generator (no augmentation) ----------------------
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            validation_split=0.20
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # ---- Training generator (with optional augmentation) -------------
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                validation_split=0.20,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )