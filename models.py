from collections import namedtuple, UserDict, Counter, defaultdict

from processing import convert_to_pattern, clear_diacritics, merge_diacritics, extract_diacritics

FakeState = namedtuple('FakeState', 'name')


class DefaultStateDict(UserDict):

    def __missing__(self, key):
        return FakeState(name=convert_to_pattern(key))


class MostFrequentPatternDiacritizer:
    """
    Diacritizer that choose the most frequent diacritization for the pattern of the word. If not found, the word is left
    undiacritized.
    """

    def __init__(self, diacritized_word_sequences):
        """
        Construct a diacritizer and populate it by the words of the diacritized sentences.
        :param diacritized_word_sequences: list of sentences where the type of every sentence is list of words.
        """
        assert isinstance(diacritized_word_sequences, list) and \
               all(isinstance(s, list) and isinstance(w, str) for s in diacritized_word_sequences for w in s)
        diacritized_words = []
        for d_sequence in diacritized_word_sequences:
            diacritized_words.extend(d_sequence)
        words_diacs = defaultdict(Counter)
        for d_w in diacritized_words:
            words_diacs[clear_diacritics(d_w)][convert_to_pattern(d_w)] += 1
        words_top_diac = {w: max(words_diacs[w].keys(), key=words_diacs[w].get) for w in words_diacs.keys()}
        self.word_diacritization = DefaultStateDict()
        self.word_diacritization.update({word: FakeState(name=diac) for word, diac in words_top_diac.items()})

    def predict(self, undiacritized_words_sequence):
        assert isinstance(undiacritized_words_sequence, list) and all(isinstance(w, str) for w in
                                                                      undiacritized_words_sequence)
        return [merge_diacritics(u_w, extract_diacritics(self.word_diacritization[u_w].name))
                for u_w in undiacritized_words_sequence]
