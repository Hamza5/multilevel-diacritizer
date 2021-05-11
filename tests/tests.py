import unittest
from pathlib import Path
from argparse import Namespace

import tensorflow as tf
import numpy as np

from multilevel_diacritizer.multi_level_diacritizer import (
    get_loaded_model, get_dataset_from, diacritize_text, get_sentences
)
from multilevel_diacritizer.model import MultiLevelDiacritizer
from multilevel_diacritizer.constants import (
    DEFAULT_WINDOW_SIZE, DEFAULT_LSTM_SIZE, DEFAULT_DROPOUT_RATE, DEFAULT_EMBEDDING_SIZE, DEFAULT_PARAMS_DIR,
    DEFAULT_SLIDING_STEP, DEFAULT_BATCH_SIZE, DIACRITICS_PATTERN
)


class ModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.u_text = ''''
        رواق - المنصة العربية للتعليم المفتوح
        مواد أكاديمية مجانية باللغة العربية في شتى المجالات والتخصصات
        المحاضرين الجامعيين، الأكاديميين، الخبراء المهنيين
        إذا كنت تقوم بتدريس أو تدريب طلاب في معهد، كلية، جامعة، منظمة،
        في تخصص معين ولديك الرغبة في نقل المحتوى التعليمي إلى جمهور أوسع، فنحن في رواق نوفر لك هذه الميزة.
        '''
        cls.data_file_path = Path('tests/train_mini.txt')
        cls.model_args = Namespace(
            window_size=DEFAULT_WINDOW_SIZE, lstm_size=DEFAULT_LSTM_SIZE, dropout_rate=DEFAULT_DROPOUT_RATE,
            embedding_size=DEFAULT_EMBEDDING_SIZE, calculate_der=False, calculate_wer=False,
            params_dir=DEFAULT_PARAMS_DIR, sliding_step=DEFAULT_SLIDING_STEP, batch_size=DEFAULT_BATCH_SIZE,
        )

    def test_loading_model(self):
        model, model_path = get_loaded_model(self.model_args)
        self.assertIsInstance(model_path, Path)
        self.assertIsInstance(model, MultiLevelDiacritizer)

    def test_train_dataset(self):
        data = get_dataset_from([self.data_file_path], self.model_args)
        self.assertIn('dataset', data.keys())
        self.assertIn('size', data.keys())
        self.assertIsInstance(data['dataset'], tf.data.Dataset)
        self.assertEqual(np.ndim(data['size']), 0)

    def test_sentences(self):
        sentences = get_sentences(self.u_text)
        self.assertIsInstance(sentences, list)
        for s in sentences:
            self.assertIsInstance(s, str)

    def test_diacritize_text(self):
        model, model_path = get_loaded_model(self.model_args)
        d_text = diacritize_text(model, self.model_args, self.u_text)
        self.assertIsInstance(d_text, str)
        self.assertEqual(DIACRITICS_PATTERN.sub('', d_text), self.u_text)
        self.assertNotEqual(d_text, self.u_text)


if __name__ == '__main__':
    unittest.main()
