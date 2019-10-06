import re
# The prefixes and suffixes that should be separated from the stem before converting it to a pattern
SEPARATED_PREFIXES = {'ل', 'ك', 'ف', 'ت', 'ب', 'ن', 'ال', 'لل'}
SEPARATED_PREFIXES.update({x + 'ال' for x in {'ك', 'ف', 'و', 'ب'}})
SEPARATED_PREFIXES.update({'س' + x for x in {'ي', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'و' + x for x in {'بال', 'كال', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'ف' + x for x in {'بال', 'كال', 'ي', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({x + 'ست' for x in {'ي', 'ا', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'و' + x + 'ست' for x in {'ي', 'ا', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'ف' + x + 'ست' for x in {'ي', 'ا', 'أ', 'ت', 'ن'}})
SEPARATED_PREFIXES.update({'باست'})
SEPARATED_SUFFIXES = {'ت', 'ك', 'ه', 'ها', 'ون', 'ين', 'ان', 'كم', 'هم', 'كن', 'هن', 'كما', 'هما'}
SEPARATED_SUFFIXES.update({'ت' + x for x in {'ه', 'ك', 'ين', 'ان', 'كم', 'هم', 'كن', 'هن', 'كما', 'هما', 'ما', 'ن'}})
MIN_STEM_LEN = 2
ORDINARY_ARABIC_LETTERS_PATTERN = re.compile(r'[بتثجحخدذرزسشصضطظعغفقكلمنه]')
DIACRITICS = set(chr(code) for code in range(0x064B, 0x0653))
DIACRITICS_PATTERN = re.compile('['+''.join(DIACRITICS)+']')


def separate_affixes(u_word):
    assert isinstance(u_word, str)
    possible_prefixes = sorted([s for s in SEPARATED_PREFIXES if u_word.startswith(s)], key=len, reverse=True)
    possible_suffixes = sorted([s for s in SEPARATED_SUFFIXES if u_word.endswith(s)], key=len, reverse=True)
    possible_prefixes.append('')
    possible_suffixes.append('')
    acceptable_affixes = []
    for p in possible_prefixes:
        for s in possible_suffixes:
            if len(u_word) - (len(p) + len(s)) >= MIN_STEM_LEN:
                acceptable_affixes.append((p, s))
    prefix, suffix = max(acceptable_affixes, key=lambda x: len(x[0]) + len(x[1]))
    return prefix, u_word[len(prefix):len(u_word)-len(suffix)], suffix


def convert_to_pattern(u_word) -> str:
    assert isinstance(u_word, str)
    prefix, stem, suffix = separate_affixes(u_word)
    stem = ORDINARY_ARABIC_LETTERS_PATTERN.sub('ح', stem)
    return prefix + stem + suffix


def clear_diacritics(d_sentence):
    return DIACRITICS_PATTERN.sub('', d_sentence)
