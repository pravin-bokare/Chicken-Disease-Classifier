import tensorflow as tf
from src.cnnClassifier.utils import save_json, read_yaml
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
import os
from from_root import from_root
from pathlib import Path


class EvaluationConfig:
    config = read_yaml(os.path.join(from_root(), CONFIG_FILE_PATH))
    params = read_yaml(os.path.join(from_root(), PARAMS_FILE_PATH))
    path_of_model = os.path.join(from_root(), config.training.trained_model_path)
    training_data = os.path.join(from_root(), config.data_ingestion.unzip_dir)
    img_size = params.IMAGE_SIZE
    batch_size = params.BATCH_SIZE


class Evaluation:
    def __init__(self):
        self.evaluation_config = EvaluationConfig()


    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.evaluation_config.img_size[:-1],
            batch_size=self.evaluation_config.batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.evaluation_config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path):
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.evaluation_config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path(os.path.join(from_root(), "scores.json")), data=scores)