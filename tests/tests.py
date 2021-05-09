import unittest
import subprocess
from pathlib import Path
import sys
print(sys.path)

PYTHON = 'python'
SCRIPT = 'multilevel_diacritizer/multi_level_diacritizer.py'
DATA_FILENAME = 'tests/train_mini.txt'


class CommandLineTestCase(unittest.TestCase):

    def test_help(self):
        try:
            p = subprocess.run([PYTHON, SCRIPT], capture_output=True, check=True, text=True)
            self.assertEqual(p.returncode, 0)
            self.assertIn('usage:', p.stdout)
            self.assertIn('--help', p.stdout)
            p = subprocess.run([PYTHON, SCRIPT, '-h'], capture_output=True, check=True, text=True)
            self.assertEqual(p.returncode, 0)
            self.assertIn('usage:', p.stdout)
            self.assertIn('--help', p.stdout)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise e

    def test_train(self):
        last_epoch_file = Path('params/last_epoch.txt')
        for file in Path('params/').glob('*'):
            file.rename(f'{file}~')
        try:
            p = subprocess.run([PYTHON, SCRIPT, 'train', '-t', DATA_FILENAME, '-v', DATA_FILENAME, '-e', '1', '-b',
                                '256'], capture_output=True, check=True, text=True, encoding='UTF-8')
            self.assertTrue(last_epoch_file.exists())
            self.assertIn('Training finished', p.stderr)
            self.assertIn('val_loss', p.stdout)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise e
        finally:
            for backup_file in Path('params/').glob('*~'):
                destination_file = Path(str(backup_file).rstrip('~'))
                if destination_file.exists():
                    destination_file.unlink()
                backup_file.rename(destination_file)
            if last_epoch_file.exists():
                last_epoch_file.unlink()

    def test_diacritization(self):
        original_file = Path(DATA_FILENAME)
        tiny_file = original_file.with_name('_tiny.'.join(original_file.name.split('.')))
        predicted_file = tiny_file.with_name('-pred.'.join(original_file.name.split('.')))
        try:
            tiny_text = '\n'.join(original_file.read_text(encoding='UTF-8').splitlines()[:100])
            tiny_file.write_text(tiny_text, encoding='UTF-8')
            p = subprocess.run([PYTHON, SCRIPT, 'diacritization', '-f', str(tiny_file), '-o', str(predicted_file)],
                               capture_output=True, check=True, text=True, encoding='UTF-8')
            self.assertIn('Model weights loaded', p.stderr)
            self.assertIn('Done', p.stderr)
            predicted_lines = predicted_file.read_text(encoding='UTF-8').splitlines()
            tiny_lines = tiny_text.splitlines()
            self.assertEqual(len(predicted_lines), len(tiny_lines))
            # The model should diacritize one sentence at least perfectly.
            self.assertGreater(len(set(predicted_lines).intersection(set(tiny_lines))), 0)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise e
        finally:
            if tiny_file.exists():
                tiny_file.unlink()
            if predicted_file.exists():
                predicted_file.unlink()


if __name__ == '__main__':
    unittest.main()
