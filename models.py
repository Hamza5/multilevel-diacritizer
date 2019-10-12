from collections import namedtuple, UserDict, Counter, defaultdict

import numpy as np
from pomegranate import HiddenMarkovModel, DiscreteDistribution

from processing import convert_to_pattern, clear_diacritics, merge_diacritics, extract_diacritics

FakeState = namedtuple('FakeState', 'name')
UNKNOWN = '<unk>'


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
        """
        Predict the sequence of diacritized words corresponding to the given undiacritized sequence.
        :param undiacritized_words_sequence: list of undiacritized words of a sentence.
        :return: list of diacritized words.
        """
        assert isinstance(undiacritized_words_sequence, list) and all(isinstance(w, str) for w in
                                                                      undiacritized_words_sequence)
        return [merge_diacritics(u_w, extract_diacritics(self.word_diacritization[u_w].name))
                for u_w in undiacritized_words_sequence]


class BigramHMMPatternDiacritizer:
    """
    Diacritizer that choose the most probable diacritization for the pattern of the word using a first order HMM.
    If a word is not found, it is left undiacritized.
    """

    def __init__(self, diacritized_word_sequences, epsilon=0.01):
        """
        Construct a diacritizer and populate it by the transitions and emission probabilities of the words of the
        diacritized sentences.
        :param diacritized_word_sequences: list of sentences where the type of every sentence is list of words.
        :param epsilon: a small strictly positive number to replace the zero probabilities.
        """
        assert isinstance(diacritized_word_sequences, list) and \
               all(isinstance(s, list) and isinstance(w, str) for s in diacritized_word_sequences for w in s)
        assert 0 < epsilon < 1
        d_words = []
        d_patterns_bigrams = []
        self.u_patterns = set()  # Needed for checking the out-of-vocabulary patterns in predict function.
        print('Extracting the words and generating the required forms...')
        for d_words_sequence in diacritized_word_sequences:
            d_words.extend(d_words_sequence)
            self.u_patterns.update([convert_to_pattern(clear_diacritics(d_w)) for d_w in d_words])
            for d_w1, d_w2 in zip(d_words_sequence[:-1], d_words_sequence[1:]):
                d_patterns_bigrams.append((convert_to_pattern(d_w1), convert_to_pattern(d_w2)))
        print('Indexing...')
        patterns_unigrams_counter = Counter([convert_to_pattern(d_w) for d_w in d_words])  # d_pattern: count.
        patterns_bigrams_counter = Counter(d_patterns_bigrams)  # (d_pattern1, d_pattern2): count.
        d_pattern_u_pattern_counter = defaultdict(Counter)  # (d_pattern, u_pattern): count.
        for d_word in d_words:
            d_pattern_u_pattern_counter[convert_to_pattern(d_word)][convert_to_pattern(clear_diacritics(d_word))] += 1
        patterns_start_counter = Counter([s[0] for s in diacritized_word_sequences])
        states = []  # Patterns.
        distributions = []  # Emissions probabilities.
        print('Calculating the probabilities...')
        for pattern, u_pattern_count in d_pattern_u_pattern_counter.items():
            u_patterns_emissions = {UNKNOWN: epsilon}
            u_patterns_emissions.update({u_pattern: count/sum(u_pattern_count.values()) - epsilon/len(u_pattern_count)
                                         for u_pattern, count in u_pattern_count.items()})
            emissions_distribution = DiscreteDistribution(u_patterns_emissions)
            states.append(pattern)
            distributions.append(emissions_distribution)
        transitions = np.ones((len(states), len(states))) * epsilon  # Initialize with a small strictly positive number.
        for i, d_pattern1 in enumerate(states):
            for j, (d_pattern2, distribution) in enumerate(zip(states, distributions)):
                if (d_pattern1, d_pattern2) in patterns_bigrams_counter.keys():
                    transitions[i, j] += patterns_bigrams_counter[d_pattern1, d_pattern2] /\
                                         patterns_unigrams_counter[d_pattern1]  # Add the true probability of the pair.
        transitions /= np.sum(transitions, axis=-1, keepdims=True)  # Normalize everything.
        transitions -= epsilon / transitions.shape[0]  # Remove the probabilities of the end state.
        end_probs = (1 - np.sum(transitions, axis=-1, keepdims=True)).flatten()
        start_probs = np.ones(transitions.shape[0]) * epsilon
        for i, state in enumerate(states):
            start_probs[i] += patterns_start_counter[state] / sum(patterns_start_counter.values())
        start_probs /= np.sum(start_probs)
        print('Building the HMM...')
        self.model = HiddenMarkovModel.from_matrix(transitions, distributions, start_probs, end_probs, states,
                                                   self.__class__.__name__, merge='None')

    def predict(self, undiacritized_words_sequence):
        """
        Predict the sequence of diacritized words corresponding to the given undiacritized sequence.
        :param undiacritized_words_sequence: list of undiacritized words of a sentence.
        :return: list of diacritized words.
        """
        assert isinstance(undiacritized_words_sequence, list) and all(isinstance(w, str) for w in
                                                                      undiacritized_words_sequence)
        sequence = []
        for u_w in undiacritized_words_sequence:
            u_p = convert_to_pattern(u_w)
            if u_p not in self.u_patterns:
                sequence.append(UNKNOWN)
            else:
                sequence.append(u_p)
        predicted_sequence = [state.name for num, state in self.model.viterbi(sequence)[1][1:-1]]
        for i in range(len(predicted_sequence)):
            if sequence[i] == UNKNOWN:
                predicted_sequence[i] = undiacritized_words_sequence[i]
            else:
                predicted_sequence[i] = merge_diacritics(undiacritized_words_sequence[i],
                                                         extract_diacritics(predicted_sequence[i]))
        return predicted_sequence
