import unittest
from io import StringIO
from contextlib import redirect_stdout
from logging import getLogger, StreamHandler

from multilevel_diacritizer.multi_level_diacritizer import (
    train_command, train_parser, diacritization_command, diacritization_parser

)
from multilevel_diacritizer.constants import DIACRITICS_PATTERN


class CommandsTestCase(unittest.TestCase):

    def setUp(self):
        self.stdout = StringIO()
        self.logging_stream = StringIO()
        getLogger('multilevel_diacritizer.multi_level_diacritizer').addHandler(StreamHandler(self.logging_stream))

    def test_train(self):
        train_model_args = train_parser.parse_args(['-t', 'tests/train_mini.txt', '-v', 'tests/train_mini.txt',
                                                    '-e', '2', '-b', '256'])
        # Back-up the existing files in PARAMS_DIR.
        for file_path in train_model_args.params_dir.glob('*'):
            file_path.rename(file_path.with_suffix(file_path.suffix + '~'))
        with redirect_stdout(self.stdout):
            train_command(train_model_args)
        self.assertIn('val_loss', self.stdout.getvalue())
        self.assertIn('Training finished', self.logging_stream.getvalue())
        self.assertIn('Random weights', self.logging_stream.getvalue())
        # Remove the generated files and restore the original ones in PARAMS_DIR.
        for file_path in train_model_args.params_dir.iterdir():
            if file_path.suffix.endswith('~'):
                destination_path = file_path.with_suffix(file_path.suffix.rstrip('~'))
                if destination_path.exists():
                    destination_path.unlink()
                file_path.rename(destination_path)
            else:
                file_path.unlink()

    def test_diacritization(self):
        args = diacritization_parser.parse_args([])
        with open('tests/train_mini.txt', encoding='UTF-8') as data_file:
            u_text = DIACRITICS_PATTERN.sub('', ''.join(data_file.readlines()[:50]))  # Take the first 50 lines.
        args.file = StringIO(u_text)
        args.out_file = StringIO()
        _close = args.out_file.close
        args.out_file.close = lambda: None  # Prevent closing the stream by the command
        diacritization_command(args)
        self.assertIn('Done', self.logging_stream.getvalue())
        self.assertIn('Model weights loaded', self.logging_stream.getvalue())
        self.assertNotEqual(args.out_file.getvalue(), u_text)
        self.assertEqual(DIACRITICS_PATTERN.sub('', args.out_file.getvalue()), u_text)
        _close()


if __name__ == '__main__':
    unittest.main()
