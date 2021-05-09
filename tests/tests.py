import unittest
from pathlib import Path
from argparse import Namespace

from multilevel_diacritizer.multi_level_diacritizer import get_loaded_model
from multilevel_diacritizer.model import MultiLevelDiacritizer
from multilevel_diacritizer.constants import (
    DEFAULT_WINDOW_SIZE, DEFAULT_LSTM_SIZE, DEFAULT_DROPOUT_RATE, DEFAULT_EMBEDDING_SIZE, DEFAULT_PARAMS_DIR,
    DEFAULT_SLIDING_STEP,
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


if __name__ == '__main__':
    unittest.main()
