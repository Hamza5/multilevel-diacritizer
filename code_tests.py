import unittest
from random import randint, choices, choice

import numpy as np

from models import BigramHMMPatternDiacritizer
from processing import *


class ProcessingFunctionsTestCase(unittest.TestCase):

    def setUp(self):
        self.arabic_letters_and_space = [chr(x) for x in list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)) +
                                         [ord(' ')]]
        high_weight_for_space = np.ones(len(self.arabic_letters_and_space))
        high_weight_for_space[-1] *= 10
        high_weight_for_space /= np.sum(high_weight_for_space)
        self.random_arabic_text = ''.join(choices(self.arabic_letters_and_space, k=1000, weights=high_weight_for_space))

    def test_clear_diacritics(self):
        self.assertEqual(clear_diacritics('أوَائِلَ الأشياء: الصُّبْحُ أوَّلُ النَّهارِ، الْوَسْمِيُّ أوَّلُ المَطرِ.'),
                         'أوائل الأشياء: الصبح أول النهار، الوسمي أول المطر.')
        self.assertEqual(clear_diacritics('Python 3.7'), 'Python 3.7')

    def test_extract_diacritics(self):
        self.assertEqual(
            extract_diacritics('الدَّعْوَى: اسم ما يُدَّعَى.'),
            ['', '', ('ّ', 'َ'), 'ْ', 'َ', '', '', '', '', '', '', '', '', '', '', 'ُ', ('ّ', 'َ'), 'َ', '', '']
        )
        random_text_without_diacritics = ''.join(choices(self.arabic_letters_and_space, k=1000))
        self.assertEqual(len(extract_diacritics(random_text_without_diacritics)), len(random_text_without_diacritics))

    def test_merge_diacritics(self):
        self.assertEqual(
            merge_diacritics(
                'الدعوى: اسم ما يدعى.',
                ['', '', ('ّ', 'َ'), 'ْ', 'َ', '', '', '', '', '', '', '', '', '', '', 'ُ', ('ّ', 'َ'), 'َ', '', '']
            ), 'الدَّعْوَى: اسم ما يُدَّعَى.'
        )

    def test_merge_extract_clear_diacritics(self):
        random_diacritics = choices(list(DIACRITICS) + [''] + [('ّ', x) for x in DIACRITICS.difference({'ّ'})],
                                    k=len(self.random_arabic_text))
        random_diacritized_text = merge_diacritics(self.random_arabic_text, random_diacritics)
        self.assertEqual(extract_diacritics(random_diacritized_text), random_diacritics)
        random_diacritized_text = self.random_arabic_text
        i = 1
        while i < len(random_diacritized_text):
            random_diacritized_text = random_diacritized_text[:i] + choice(list(DIACRITICS)) + \
                                      random_diacritized_text[i:]
            i += randint(2, 10)
        self.assertEqual(merge_diacritics(clear_diacritics(random_diacritized_text),
                                          extract_diacritics(random_diacritized_text)),
                         random_diacritized_text)

    def test_separate_affixes(self):
        if ' ' in self.random_arabic_text:
            self.assertRaises(AssertionError, separate_affixes, self.random_arabic_text)
        for word in self.random_arabic_text.split(' '):
            prefix, stem, suffix = separate_affixes(word)
            self.assertIn(prefix, SEPARATED_PREFIXES.union({''}))
            self.assertIn(suffix, SEPARATED_SUFFIXES.union({''}))
            self.assertEqual(prefix + stem + suffix, word)
            self.assertRaises(AssertionError, separate_affixes, word+choice(list(DIACRITICS)))

    def test_convert_to_pattern(self):
        random_diacritized_text = self.random_arabic_text
        i = 1
        while i < len(random_diacritized_text):
            random_diacritized_text = random_diacritized_text[:i] + choice(list(DIACRITICS)) + \
                                      random_diacritized_text[i:]
            i += randint(2, 10)
        self.assertEqual(len(convert_to_pattern(random_diacritized_text)), len(random_diacritized_text))
        words_patterns = {
            'المعْنَى': 'الححْحَا',
            'دلالة': 'ححاحة',
            'بالوضع': 'بالوحح',
            'الجمع': 'الححح',
            'لإِدراك': 'لإِححاك',
            'والتفرقة': 'والححححة',
            'دلَّ': 'ححَّ',
            'نساء': 'نحاء',
            'صفات': 'ححات',
            'جَعَلَهُ': 'حَحَحَهُ',
            'وسُؤْرُ': 'وحُءْحُ',
            'ل': 'ح',
            'مضمون': 'حححون',
            'ورأيتهما': 'وحءيتهما',
            'فيضحكن': 'فيححكن',
            'تلعبان': 'تحححان',
            'استجمع': 'استححح',
            'أكل': 'أحح',
            'Xor': 'Xor',
            '345': '345',
            'بصفين': 'بححين'
        }
        for word, pattern in words_patterns.items():
            self.assertEqual(pattern, convert_to_pattern(word))
        self.assertEqual(convert_to_pattern(' '.join(words_patterns.keys())), ' '.join(words_patterns.values()))

    def test_convert_non_arabic(self):
        words = self.random_arabic_text.split(' ')
        non_arabic_words_count = 10
        for _ in range(non_arabic_words_count):
            random_non_arabic_word = ''.join(
                chr(x) for x in choice([choices(range(ord('0'), ord('9')+1), k=randint(1, 5)),
                                        choices(range(ord('A'), 256), k=randint(2, 10))])
            )
            words.insert(randint(0, len(words)), random_non_arabic_word)
        random_sentence = ' '.join(words)
        converted_sentence = convert_non_arabic(random_sentence)
        self.assertNotEqual(random_sentence, converted_sentence)
        self.assertEqual(converted_sentence.count(FOREIGN) + converted_sentence.count(NUMBER), non_arabic_words_count)


class HMMTestCase(unittest.TestCase):

    def setUp(self):
        self.arabic_letters = [chr(x) for x in list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B))]
        self.latin_letters = [chr(x) for x in range(ord('A'), ord('z')+1)]
        self.diacritics = list(DIACRITICS.union({''}))
        random_sentences = []
        for i in range(50):
            random_sentence = []
            for j in range(randint(2, 10)):
                word = choices(self.arabic_letters, k=randint(2, 8))
                word = ''.join([l + d for l, d in zip(word, choices(self.diacritics, k=len(word)))])
                random_sentence.append(word)
            random_sentences.append(random_sentence)
        for s in choices(random_sentences, k=5):
            s.insert(randint(0, len(s)), str(randint(0, 9999)))
            s.insert(randint(0, len(s)), ''.join(choices(self.latin_letters, k=randint(1, 5))))
        self.hmm = BigramHMMPatternDiacritizer(random_sentences)

    def test_arrays_indexes_sizes(self):
        self.assertEqual(self.hmm.transitions.shape[0], self.hmm.transitions.shape[1])
        self.assertEqual(self.hmm.transitions.shape[0], self.hmm.emissions.shape[0])
        self.assertEqual(self.hmm.priors.shape[0], self.hmm.transitions.shape[0])
        self.assertEqual(len(self.hmm.d_patterns_indexes), self.hmm.emissions.shape[0])
        self.assertEqual(len(self.hmm.u_patterns_indexes), self.hmm.emissions.shape[1])

    def test_array_sums(self):
        self.assertTrue(np.all(np.abs(np.sum(self.hmm.transitions, axis=-1) -
                                      np.ones(self.hmm.transitions.shape[0])) < 1e-4),
                        'Transitions sums:\n' + str(np.sum(self.hmm.transitions, axis=-1)) +
                        '\nExpected:\n' + str(np.ones(self.hmm.transitions.shape[0])))
        self.assertTrue(np.all(np.abs(np.sum(self.hmm.emissions, axis=-1) -
                                      np.ones(self.hmm.emissions.shape[0])) < 1e-4),
                        'Emissions sums: ' + str(np.sum(self.hmm.emissions, axis=-1)) +
                        '\nExpected:\n' + str(np.ones(self.hmm.emissions.shape[0])))
        self.assertAlmostEqual(sum(self.hmm.priors), 1, 4)


if __name__ == '__main__':
    unittest.main()
