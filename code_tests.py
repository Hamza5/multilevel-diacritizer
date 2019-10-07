import unittest
from random import randint, choices, choice

from processing import *


class ProcessingFunctionsTestCase(unittest.TestCase):

    def setUp(self):
        self.arabic_letters_and_space = [chr(x) for x in list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)) +
                                         [ord(' ')]]

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
        random_arabic_text = ''.join(choices(self.arabic_letters_and_space, k=1000))
        random_diacritics = choices(list(DIACRITICS) + [''] + [('ّ', x) for x in DIACRITICS.difference({'ّ'})],
                                    k=len(random_arabic_text))
        random_diacritized_text = merge_diacritics(random_arabic_text, random_diacritics)
        self.assertEqual(extract_diacritics(random_diacritized_text), random_diacritics)
        random_diacritized_text = random_arabic_text
        i = 1
        while i < len(random_diacritized_text):
            random_diacritized_text = random_diacritized_text[:i] + choice(list(DIACRITICS)) +\
                                      random_diacritized_text[i:]
            i += randint(2, 10)
        self.assertEqual(merge_diacritics(clear_diacritics(random_diacritized_text),
                                          extract_diacritics(random_diacritized_text)),
                         random_diacritized_text)

    def test_convert_to_pattern(self):
        random_arabic_text = ''.join(choices(self.arabic_letters_and_space, k=10))
        random_diacritized_text = random_arabic_text
        i = 1
        while i < len(random_diacritized_text):
            random_diacritized_text = random_diacritized_text[:i] + choice(list(DIACRITICS)) + \
                                      random_diacritized_text[i:]
            i += randint(2, 10)
        self.assertEqual(len(convert_to_pattern(random_diacritized_text)), len(random_diacritized_text))
        words_patterns = {
            'المعْنَى': 'الححْحَى',
            'دلالة': 'ححاحة',
            'بالوضع': 'بالوحح',
            'الجمع': 'الححح',
            'لإِدراك': 'لإِححاك',
            'والتفرقة': 'والححححة',
            'دلَّ': 'ححَّ',
            'نساء': 'نحاء',
            'صفات': 'ححات',
            'جَعَلَهُ': 'حَحَحَهُ',
            'وسُؤْرُ': 'وحُؤْحُ',
            'ل': 'ح',
            'مضمون': 'مححون',
            'ورأيتهما': 'وحأيتهما',
            'فيضحكن': 'فيححكن',
            'تلعبان': 'تحححان',
        }
        for word, pattern in words_patterns.items():
            self.assertEqual(convert_to_pattern(word), pattern)


if __name__ == '__main__':
    unittest.main()
