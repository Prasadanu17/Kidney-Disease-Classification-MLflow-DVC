import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger


class Evaluation:
    """Model evaluation component.

    Loads the trained model, runs inference on the validation split,
    computes classification metrics (accuracy, precision, recall, f1, roc-auc,
    confusion matrix), saves them to JSON, and logs everything to MLflow /
    DagsHub.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _valid_generator(self):
        """Build a validation ImageDataGenerator using the same 30 % split
        that the research notebook uses for evaluation."""

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # (H, W)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,       # must be False for reliable label alignment
            **dataflow_kwargs
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            logger.info(f"Standard load_model failed ({e}), loading with compile=False and recompiling...")
            model = tf.keras.models.load_model(path, compile=False)
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluation(self):
        """Run full evaluation: keras metrics + sklearn classification metrics."""

        logger.info("Loading trained model …")
        self.model = self.load_model(self.config.path_of_model)

        logger.info("Building validation generator …")
        self._valid_generator()

        # ---- Keras built-in evaluation (loss + accuracy) ---------------
        logger.info("Running model.evaluate() …")
        self.score = self.model.evaluate(self.valid_generator, verbose=1)
        keras_loss      = float(self.score[0])
        keras_accuracy  = float(self.score[1])

        # ---- Full predictions for sklearn metrics ----------------------
        logger.info("Collecting predictions for sklearn metrics …")
        self.valid_generator.reset()

        y_pred_proba = self.model.predict(self.valid_generator, verbose=1)
        # class_indices maps class-name -> integer index (alphabetical order)
        class_indices = self.valid_generator.class_indices
        logger.info(f"Class indices: {class_indices}")

        y_true = self.valid_generator.classes          # integer labels
        y_pred = np.argmax(y_pred_proba, axis=1)       # predicted class ints

        n_classes = y_pred_proba.shape[1]

        # ---- Classification metrics ------------------------------------
        accuracy  = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred,
                                          average="binary" if n_classes == 2
                                          else "weighted",
                                          zero_division=0))
        recall    = float(recall_score(y_true, y_pred,
                                       average="binary" if n_classes == 2
                                       else "weighted",
                                       zero_division=0))
        f1        = float(f1_score(y_true, y_pred,
                                   average="binary" if n_classes == 2
                                   else "weighted",
                                   zero_division=0))

        # ROC-AUC – works for binary (probability of positive class)
        # For multi-class use ovr macro; guard against a single-class batch
        try:
            if n_classes == 2:
                roc_auc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class="ovr", average="macro"
                ))
        except ValueError as exc:
            logger.warning(f"ROC-AUC could not be computed: {exc}")
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()

        # Classification report (for logging)
        report = classification_report(
            y_true, y_pred,
            target_names=list(class_indices.keys()),
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")

        # Store for later use in log_into_mlflow
        self.metrics = {
            "loss":      keras_loss,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1,
        }
        if roc_auc is not None:
            self.metrics["roc_auc"] = roc_auc

        self.confusion_matrix = cm

        logger.info(f"Evaluation metrics: {self.metrics}")
        logger.info(f"Confusion matrix:   {cm}")

        self.save_score()

    def save_score(self):
        """Persist metrics + confusion matrix to artifacts/evaluation/scores.json."""

        output = dict(self.metrics)
        output["confusion_matrix"] = self.confusion_matrix

        save_json(
            path=Path("artifacts/evaluation/scores.json"),
            data=output
        )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    def log_into_mlflow(self):
        """Log params, metrics, confusion-matrix artifact and model to MLflow."""

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
        except Exception as e:
            logger.warning(f"Could not set registry URI: {e}")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            # Log all training hyper-parameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metrics(self.metrics)

            # Save confusion matrix as a local JSON artifact
            cm_path = Path("artifacts/evaluation/confusion_matrix.json")
            save_json(path=cm_path, data={"confusion_matrix": self.confusion_matrix})
            mlflow.log_artifact(str(cm_path))

            # Register / log the model
            try:
                if tracking_url_type_store != "file":
                    try:
                        mlflow.keras.log_model(
                            self.model,
                            "model",
                            registered_model_name="VGG16Model"
                        )
                    except Exception as reg_err:
                        logger.warning(f"Model registration failed ({reg_err}); logging model without registration.")
                        mlflow.keras.log_model(self.model, "model")
                else:
                    mlflow.keras.log_model(self.model, "model")
            except Exception as model_err:
                logger.warning(f"Logging model to MLflow failed: {model_err}")
