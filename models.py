from collections import namedtuple, UserDict, Counter, defaultdict

import numpy as np
from pomegranate import HiddenMarkovModel, DiscreteDistribution, State

from processing import convert_to_pattern, clear_diacritics, merge_diacritics, extract_diacritics

FakeState = namedtuple('FakeState', 'name')
UNKNOWN = '<unk>'
START = '<s>'
END = '</s>'


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


class BigramHMMPatternDiacritizer(HiddenMarkovModel):
    """
    Diacritizer that choose the most probable diacritization for the pattern of the word using an HMM with one step look
    behind. If not found, the word is left undiacritized.
    """

    def __init__(self, diacritized_word_sequences, epsilon=0.01):
        super(BigramHMMPatternDiacritizer, self).__init__('BigramHMMPatternDiacritizer')
        d_words = []
        d_patterns_bigrams = []
        self.u_words = set()
        print('Extracting the words and generating the required forms...')
        for d_words_sequence in diacritized_word_sequences:
            d_words_sequence = [START] + d_words_sequence + [END]
            d_words.extend(d_words_sequence)
            u_words = [clear_diacritics(d_w) for d_w in d_words]
            self.u_words.update(u_words)
            for d_w1, d_w2 in zip(d_words_sequence[:-1], d_words_sequence[1:]):
                d_patterns_bigrams.append((convert_to_pattern(d_w1), convert_to_pattern(d_w2)))
        print('Indexing...')
        patterns_unigrams_counter = Counter([convert_to_pattern(d_w) for d_w in d_words])
        patterns_bigrams_counter = Counter(d_patterns_bigrams)
        d_pattern_u_word_counter = defaultdict(Counter)
        for d_word in d_words:
            d_pattern_u_word_counter[convert_to_pattern(d_word)][clear_diacritics(d_word)] += 1
        states_distributions = {}
        print('Calculating the probabilities...')
        for pattern, u_word_count in d_pattern_u_word_counter.items():
            if pattern not in (START, END):
                u_words_emissions = {UNKNOWN: epsilon}
                u_words_emissions.update({u_word: count/sum(u_word_count.values()) - epsilon/len(u_word_count)
                                          for u_word, count in u_word_count.items()})
            else:
                u_words_emissions = {u_word: 1 for u_word in u_word_count.keys()}
            emissions_distribution = DiscreteDistribution(u_words_emissions)
            states_distributions[pattern] = State(emissions_distribution, name=pattern)
        self.start = states_distributions[START]
        self.end = states_distributions[END]
        self.add_states(list(states_distributions.values()))
        for pattern1 in states_distributions.keys():
            states = []
            trans_probs = np.ones(len(states_distributions)) * epsilon
            for i, (pattern2, state) in enumerate(states_distributions.items()):
                states.append(state)
                if (pattern1, pattern2) in patterns_bigrams_counter.keys():
                    trans_probs[i] += patterns_bigrams_counter[pattern1, pattern2] / patterns_unigrams_counter[pattern1]
            trans_probs /= trans_probs.sum()
            self.add_transitions(states_distributions[pattern1], states, trans_probs.tolist())
        print('Building the HMM...')
        self.bake(verbose=True)

    def predict(self, undiacritized_words_sequence):
        assert isinstance(undiacritized_words_sequence, list) and all(isinstance(w, str) for w in
                                                                      undiacritized_words_sequence)
        sequence = []
        for u_w in undiacritized_words_sequence:
            if u_w not in self.u_words:
                sequence.append(UNKNOWN)
            else:
                sequence.append(u_w)
        predicted_sequence = [state.name for num, state in
                              super(BigramHMMPatternDiacritizer, self).viterbi(sequence + [END])[1][1:-1]]
        for i in range(len(predicted_sequence)):
            if sequence[i] == UNKNOWN:
                predicted_sequence[i] = undiacritized_words_sequence[i]
            else:
                predicted_sequence[i] = merge_diacritics(undiacritized_words_sequence[i],
                                                         extract_diacritics(predicted_sequence[i]))
        return predicted_sequence
