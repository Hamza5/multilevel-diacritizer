import re
from typing import Iterable
# The prefixes and suffixes that should be separated from the stem before converting it to a pattern
SEPARATED_PREFIXES = {'ل', 'ك', 'ف', 'ت', 'ب', 'ن', 'ال', 'لل', 'أ'}
SEPARATED_PREFIXES.update({x + 'ال' for x in {'ك', 'ف', 'و', 'ب'}})
SEPARATED_PREFIXES.update({'س' + x for x in {'ي', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'و' + x for x in {'بال', 'كال', 'ت', 'أ', 'ن'}})
SEPARATED_PREFIXES.update({'ف' + x for x in {'بال', 'كال', 'ي', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({x + 'ست' for x in {'ي', 'ا', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'و' + x + 'ست' for x in {'ي', 'ا', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'ف' + x + 'ست' for x in {'ي', 'ا', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'باست'})
SEPARATED_SUFFIXES = {'ت', 'ك', 'ه', 'ها', 'ون', 'ين', 'ان', 'كم', 'هم', 'كن', 'هن', 'كما', 'هما'}
SEPARATED_SUFFIXES.update({'ت' + x for x in {'ه', 'ك', 'ين', 'ان', 'كم', 'هم', 'كن', 'هن', 'كما', 'هما', 'ها', 'ما',
                                             'ن'}})
MIN_STEM_LEN = 2
ORDINARY_ARABIC_LETTERS_PATTERN = re.compile(r'[بتثجحخدذرزسشصضطظعغفقكلمنه]')
HAMZAT_PATTERN = re.compile(r'[ءأآؤئ]')
DIACRITICS = set(chr(code) for code in range(0x064B, 0x0653))
DIACRITICS_PATTERN = re.compile('['+''.join(DIACRITICS)+']')
NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
ARABIC_NUMBER_PATTERN = re.compile('((?:['+''.join(ARABIC_LETTERS)+']['+''.join(DIACRITICS)+r']*)+|\d+(?:\.\d+)?)')
SENTENCE_SEPARATORS = ';,،؛.:؟!'
PUNCTUATION = SENTENCE_SEPARATORS + '۩﴿﴾«»ـ' +\
              ''.join([chr(x) for x in range(0x0021, 0x0030)]+[chr(x) for x in range(0x003A, 0x0040)] +
                      [chr(x) for x in range(0x005B, 0x0060)]+[chr(x) for x in range(0x007B, 0x007F)])
PUNCTUATION_PATTERN = re.compile('['+''.join(PUNCTUATION)+']+')
NUMBER = '0'
FOREIGN = '<FOR>'


def separate_affixes(u_word: str) -> (str, str, str):
    """
    Separate some affixes from the stem of an undiacritized word and return all the three parts.
    :param u_word: str, an undiacritized word.
    :return: (str, str, str), prefix, stem and suffix.
    """
    assert isinstance(u_word, str) and set(u_word).intersection(DIACRITICS) == set()
    possible_prefixes = sorted([s for s in SEPARATED_PREFIXES if u_word.startswith(s)], key=len, reverse=True)
    possible_suffixes = sorted([s for s in SEPARATED_SUFFIXES if u_word.endswith(s)], key=len, reverse=True)
    possible_prefixes.append('')
    possible_suffixes.append('')
    acceptable_affixes = []
    for p in possible_prefixes:
        for s in possible_suffixes:
            if len(u_word) - (len(p) + len(s)) >= MIN_STEM_LEN:
                acceptable_affixes.append((p, s))
    if len(acceptable_affixes) == 0:
        acceptable_affixes = [('', '')]
    prefix, suffix = max(acceptable_affixes, key=lambda x: len(x[0]) + len(x[1]))
    return prefix, u_word[len(prefix):len(u_word)-len(suffix)], suffix


def clear_diacritics(d_text: str) -> str:
    """
    Remove all standard diacritics from the text.
    :param d_text: str, the diacritized text.
    :return: str, the text undiacritized.
    """
    return DIACRITICS_PATTERN.sub('', d_text)


def extract_diacritics(d_text: str) -> list:
    """
    Return the diacritics from the text while keeping their original positions including the Shaddah marks.
    :param d_text: str, the diacritized text.
    :return: list, the diacritics. Positions with double diacritics have a tuple as elements.
    """
    assert isinstance(d_text, str)
    diacritics = []
    for i in range(1, len(d_text)):
        if d_text[i] in DIACRITICS:
            if d_text[i - 1] == 'ّ':
                diacritics[-1] = (d_text[i - 1], d_text[i])
            else:
                diacritics.append(d_text[i])
        elif d_text[i - 1] not in DIACRITICS:
            diacritics.append('')
    if d_text[-1] not in DIACRITICS:
        diacritics.append('')
    return diacritics


def merge_diacritics(u_text: str, diacritics: Iterable) -> str:
    """
    Return the diacritized text resulted from merging the text with the diacritics extracted by separate_diacritics.
    :param u_text: str, the undiacritized text.
    :param diacritics: list of str or tuple, the diacritics. Positions with double diacritics have a tuple as elements.
    :return: str, the diacritized text.
    """
    assert isinstance(u_text, str)
    assert isinstance(diacritics, Iterable)
    return ''.join([l+''.join(d) for l, d in zip(u_text, diacritics)])


def convert_to_pattern(word: str) -> str:
    """
    Return the corresponding pattern of a given word.
    :param word: str, the word to convert.
    :return: str, the pattern of the word.
    """
    assert isinstance(word, str)
    prefix, stem, suffix = separate_affixes(clear_diacritics(word))
    stem = HAMZAT_PATTERN.sub('ء', stem)
    stem = ORDINARY_ARABIC_LETTERS_PATTERN.sub('ح', stem)
    return merge_diacritics(prefix + stem + suffix, extract_diacritics(word))


def convert_non_arabic(text) -> str:
    """
    Substitute non Arabic tokens other than spaces and punctuation with the corresponding replacements.
    :param text: str, text containing non-Arabic characters.
    :return: str, the text with non-Arabic tokens replaced.
    """
    assert isinstance(text, str)
    r = ''
    for x in ARABIC_NUMBER_PATTERN.split(text):
        if NUMBER_PATTERN.match(x):
            r += NUMBER
        elif x.isspace() or ARABIC_NUMBER_PATTERN.match(x) or PUNCTUATION_PATTERN.match(x):
            r += x
        elif x != '':
            r += FOREIGN
    return r
