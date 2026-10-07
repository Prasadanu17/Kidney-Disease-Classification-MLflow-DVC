import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(
            model,
            classes,
            freeze_all,
            freeze_till,
            learning_rate):
        """
        Build the full classification model on top of VGG16.

        Fine-tuning strategy:
        - freeze_all=True, freeze_till=None  → freeze everything (feature-extract only)
        - freeze_all=False, freeze_till=4    → unfreeze last 4 layers (block5 conv layers)
          while keeping blocks 1-4 frozen.

        VGG16 block5 layers (indices from model.layers):
          index -4  block5_conv1
          index -3  block5_conv2
          index -2  block5_conv3
          index -1  block5_pool
        Setting freeze_till=4 unfreezes these 4 layers only.
        """

        # ---------- layer freezing ----------------------------------------
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze everything except the last freeze_till layers
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
            for layer in model.layers[-freeze_till:]:
                layer.trainable = True

        # ---------- classification head -----------------------------------
        flatten_in = tf.keras.layers.Flatten()(model.output)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # ---------- parameter summary -------------------------------------
        total_params     = full_model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w)
            for w in full_model.trainable_weights
        )
        non_trainable_params = total_params - trainable_params

        print("\n" + "=" * 60)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f"  Total parameters     : {total_params:,}")
        print(f"  Trainable parameters : {trainable_params:,}")
        print(f"  Non-trainable params : {non_trainable_params:,}")
        print("=" * 60 + "\n")

        full_model.summary()

        return full_model

    def update_base_model(self):
        """
        Build the updated model with controlled fine-tuning.

        Fine-tuning decision:
        - We unfreeze the last 4 VGG16 layers (block5: conv1, conv2, conv3, pool)
          while keeping all earlier blocks (block1–block4) frozen.
        - The Dense classification head is always trainable.
        - freeze_till=4 means model.layers[-4:] are trainable.
        - Learning rate kept at the value in params.yaml (LEARNING_RATE=0.0001).
          A small LR is important for fine-tuning so we do not destroy
          the pre-trained ImageNet weights in block5.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,     # enable controlled fine-tuning
            freeze_till=4,        # unfreeze last 4 layers = VGG16 block5
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)