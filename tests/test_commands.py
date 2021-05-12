import unittest
import sys
from io import StringIO
from contextlib import redirect_stdout
from logging import getLogger, StreamHandler

from multilevel_diacritizer.multi_level_diacritizer import train_command, train_parser


class CommandsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_model_args = train_parser.parse_args(['-t', 'tests/train_mini.txt', '-v', 'tests/train_mini.txt',
                                                        '-e', '2', '-b', '256'])

    def setUp(self):
        self.stdout = StringIO()
        self.logging_stream = StringIO()
        getLogger('multilevel_diacritizer.multi_level_diacritizer').addHandler(StreamHandler(self.logging_stream))
        # Back-up the existing files in PARAMS_DIR.
        for file_path in self.train_model_args.params_dir.glob('*'):
            file_path.rename(file_path.with_suffix(file_path.suffix+'~'))

    def test_train(self):
        with redirect_stdout(self.stdout):
            train_command(self.train_model_args)
        self.assertIn('val_loss', self.stdout.getvalue())
        self.assertIn('Training finished', self.logging_stream.getvalue())

    def tearDown(self):
        # Remove the generated files and restore the original ones in PARAMS_DIR.
        for file_path in self.train_model_args.params_dir.iterdir():
            if file_path.suffix.endswith('~'):
                destination_path = file_path.with_suffix(file_path.suffix.rstrip('~'))
                if destination_path.exists():
                    destination_path.unlink()
                file_path.rename(destination_path)
            else:
                file_path.unlink()

    @classmethod
    def tearDownClass(cls):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == '__main__':
    unittest.main()
