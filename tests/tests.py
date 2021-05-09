import unittest
from pathlib import Path
from argparse import Namespace

import tensorflow as tf
import numpy as np

from multilevel_diacritizer.multi_level_diacritizer import get_loaded_model, get_dataset_from
from multilevel_diacritizer.model import MultiLevelDiacritizer
from multilevel_diacritizer.constants import (
    DEFAULT_WINDOW_SIZE, DEFAULT_LSTM_SIZE, DEFAULT_DROPOUT_RATE, DEFAULT_EMBEDDING_SIZE, DEFAULT_PARAMS_DIR,
    DEFAULT_SLIDING_STEP, DEFAULT_BATCH_SIZE
)


class ModelTestCase(unittest.TestCase):

    def test_loading_model(self):
        model, model_path = get_loaded_model(
            Namespace(window_size=DEFAULT_WINDOW_SIZE, lstm_size=DEFAULT_LSTM_SIZE, dropout_rate=DEFAULT_DROPOUT_RATE,
                      embedding_size=DEFAULT_EMBEDDING_SIZE, calculate_der=False, calculate_wer=False,
                      params_dir=DEFAULT_PARAMS_DIR, sliding_step=DEFAULT_SLIDING_STEP)
        )
        self.assertIsInstance(model_path, Path)
        self.assertIsInstance(model, MultiLevelDiacritizer)

    def test_train_dataset(self):
        data = get_dataset_from(
            [Path('tests/train_mini.txt')],
            Namespace(batch_size=DEFAULT_BATCH_SIZE, window_size=DEFAULT_WINDOW_SIZE, sliding_step=DEFAULT_SLIDING_STEP)
        )
        self.assertIn('dataset', data.keys())
        self.assertIn('size', data.keys())
        self.assertIsInstance(data['dataset'], tf.data.Dataset)
        self.assertEqual(np.ndim(data['size']), 0)


if __name__ == '__main__':
    unittest.main()
