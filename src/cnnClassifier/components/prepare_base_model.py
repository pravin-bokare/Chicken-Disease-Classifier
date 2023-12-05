import os
import tensorflow as tf
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils import read_yaml
from from_root import from_root
from from_root import from_root


class PrepareBaseModelConfig:
    config = read_yaml(os.path.join(from_root(), CONFIG_FILE_PATH))
    param = read_yaml(os.path.join(from_root(), PARAMS_FILE_PATH))
    root_dir = config.prepare_base_model.root_dir
    base_model_path = config.prepare_base_model.base_model_path
    updated_base_model_path = config.prepare_base_model.updated_base_model_path
    img_size = param.IMAGE_SIZE
    learning_rate = param.LEARNING_RATE
    include_top = param.INCLUDE_TOP
    weight = param.WEIGHTS
    classes = param.CLASSES


class PrepareBaseModel:
    def __int__(self):
        self.prepare_base_model_config = PrepareBaseModelConfig()

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.prepare_base_model_config.img_size,
            weights=self.prepare_base_model_config.weight,
            include_top=self.prepare_base_model_config.include_top
        )

        self.save_model(path=self.prepare_base_model_config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till >0):
            for layer in model.layers[:-freeze_till]:
                model.trainable=False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model=self._prepare_full_model(
            model=self.model,
            classes=self.prepare_base_model_config.classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.prepare_base_model_config.learning_rate
        )

        self.save_model(path=self.prepare_base_model_config.updated_base_model_path, model=self.full_model)


    @staticmethod
    def save_model(path, model):
        model.save(path)
