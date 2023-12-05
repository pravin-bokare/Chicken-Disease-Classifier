import os.path
from from_root import from_root
import tensorflow as tf
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils import read_yaml
from src.cnnClassifier.components.prepare_callbacks import PrepareCallbacks


class TrainingConfig:
    config = read_yaml(os.path.join(from_root(), CONFIG_FILE_PATH))
    params = read_yaml(os.path.join(from_root(), PARAMS_FILE_PATH))
    root_dir = os.path.join(from_root(), config.training.root_dir)
    trained_model_path = os.path.join(from_root(), config.training.trained_model_path)
    updated_base_model_path = os.path.join(from_root(), config.prepare_base_model.updated_base_model_path)
    training_data = os.path.join(from_root(), config.data_ingestion.unzip_dir, 'PetImages')
    epochs = params.EPOCHS
    batch_size = params.BATCH_SIZE
    augmentation = params.AUGMENTATION
    img_size = params.IMAGE_SIZE


class Training:
    def __init__(self):
        self.training_config = TrainingConfig()

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.training_config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  # Include 'rescale' here
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            batch_size=self.training_config.batch_size,
            interpolation='bilinear'
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs  # Include the dictionary here
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.training_config.training_data,
            subset='validation',
            shuffle=False,
            target_size=self.training_config.img_size[:-1],
            **dataflow_kwargs
        )

        if self.training_config.augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs  # Include the same 'rescale' argument here
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.training_config.training_data,
            subset='training',
            shuffle=True,
            target_size=self.training_config.img_size[:-1],
            **dataflow_kwargs
        )

    def save_model(self, path, model):
        model.save(path)

    def train(self, callback_list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.training_config.epochs,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(self.training_config.trained_model_path, self.model)



'''
if __name__ == '__main__':
    prepare_callbacks = PrepareCallbacks()
    callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

    training = Training()
    training.get_base_model()
    training.train_valid_generator()
    training.train(callback_list=callback_list)
'''
