import os
import tensorflow as tf
import time
from from_root import from_root
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils import read_yaml


class PrepareCallbackConfig:
    config = read_yaml(os.path.join(from_root(), CONFIG_FILE_PATH))
    params = read_yaml(os.path.join(from_root(), PARAMS_FILE_PATH))
    root_dir = os.path.join(from_root(), config.prepare_callbacks.root_dir)
    tensorboard_root_log_dir = os.path.join(from_root(), config.prepare_callbacks.tensorboard_root_log_dir)
    checkpoint_model_filepath = os.path.join(from_root(), config.prepare_callbacks.checkpoint_model_filepath)


class PrepareCallbacks:
    def __init__(self):
        self.prepare_callbacks_config = PrepareCallbackConfig()

    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(self.prepare_callbacks_config.tensorboard_root_log_dir, f'tb_logs_at_{timestamp}')
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    def _create_ckpt_callbacks(self):  # Remove the @property decorator
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.prepare_callbacks_config.checkpoint_model_filepath,
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks(),
            self._create_ckpt_callbacks()  # Call it as a method
        ]


'''
if __name__ == '__main__':
    prepare_callbacks = PrepareCallbacks()
    callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
'''