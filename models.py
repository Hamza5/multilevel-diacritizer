from collections import UserDict, Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor as PoolExecutor, wait, as_completed
from os import cpu_count

import numpy as np

from processing import convert_to_pattern, clear_diacritics, merge_diacritics, extract_diacritics, convert_non_arabic, \
    FOREIGN, NUMBER

UNKNOWN = '<unk>'
MAX_PARALLEL_RUNS = cpu_count()


class IndexDict(UserDict):
    next_index = 0

    def __setitem__(self, key, value):
        if key not in self.data.keys():
            self.data[key] = self.next_index
            self.next_index += value


class DefaultDict(UserDict):

    def __missing__(self, key):
        return convert_to_pattern(key)


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
        diacritized_word_sequences = [[convert_non_arabic(w) for w in s] for s in diacritized_word_sequences]
        diacritized_words = []
        for d_sequence in diacritized_word_sequences:
            diacritized_words.extend(d_sequence)
        words_diacs = defaultdict(Counter)
        for d_w in diacritized_words:
            words_diacs[clear_diacritics(d_w)][convert_to_pattern(d_w)] += 1
        words_top_diac = {w: max(words_diacs[w].keys(), key=words_diacs[w].get) for w in words_diacs.keys()}
        self.word_diacritization = DefaultDict()
        self.word_diacritization.update({word: diac for word, diac in words_top_diac.items()})

    def predict(self, undiacritized_words_sequence):
        """
        Predict the sequence of diacritized words corresponding to the given undiacritized sequence.
        :param undiacritized_words_sequence: list of undiacritized words of a sentence.
        :return: list of diacritized words.
        """
        assert isinstance(undiacritized_words_sequence, list) and all(isinstance(w, str) for w in
                                                                      undiacritized_words_sequence)
        return [merge_diacritics(u_w, extract_diacritics(self.word_diacritization[u_w]))
                for u_w in undiacritized_words_sequence]


class BigramHMMPatternDiacritizer:
    """
    Diacritizer that choose the most probable diacritization for the pattern of the word using a first order HMM.
    If a word is not found, it is left undiacritized.
    """

    def __init__(self, diacritized_sequences, epsilon=0.01):
        """
        Construct a diacritizer and populate it by the transitions and emission probabilities of the words of the
        diacritized sentences.
        :param diacritized_sequences: list of sentences where the type of every sentence is list of words.
        :param epsilon: a small strictly positive number to replace the zero probabilities.
        """
        assert isinstance(diacritized_sequences, list) and \
               all(isinstance(s, list) and isinstance(w, str) for s in diacritized_sequences for w in s)
        assert 0 < epsilon < 1

        print('Extracting the words and generating the required forms...')
        self.d_patterns_indexes = IndexDict({NUMBER: 1, FOREIGN: 1})
        d_words, self.u_patterns_indexes, d_patterns_bigrams = self.generate_words_patterns_bigrams(
            diacritized_sequences)

        print('Indexing...')
        d_patterns_unigrams_counter, d_patterns_bigrams_counter, d_pattern_u_pattern_counter, patterns_start_counter = \
            self.calculate_frequencies(diacritized_sequences, d_words, d_patterns_bigrams, self.d_patterns_indexes)

        print('Calculating the emissions...')
        self.emissions = self.calculate_emissions(d_pattern_u_pattern_counter, self.u_patterns_indexes,
                                                  self.d_patterns_indexes, epsilon)

        print('Calculating the transitions...')
        self.transitions = self.calculate_transitions(self.d_patterns_indexes, d_patterns_bigrams_counter,
                                                      d_patterns_unigrams_counter, epsilon)

        print('Calculating the priors...')
        self.priors = self.calculate_priors(patterns_start_counter, self.d_patterns_indexes, epsilon)
        self.d_patterns_inverted_indexes = dict((d_pattern, i) for i, d_pattern in self.d_patterns_indexes.items())

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
            u_p = convert_to_pattern(convert_non_arabic(u_w))
            if u_p not in self.u_patterns_indexes.keys():
                sequence.append(self.u_patterns_indexes[UNKNOWN])
            else:
                sequence.append(self.u_patterns_indexes[u_p])
        predicted_sequence = [self.d_patterns_inverted_indexes[i] for i in
                              viterbi(np.array(sequence, dtype=np.int), self.transitions, self.emissions, self.priors)]
        restored_sequence = []
        for i in range(len(predicted_sequence)):
            if sequence[i] in (UNKNOWN, FOREIGN, '0'):
                restored_sequence.append(undiacritized_words_sequence[i])
            else:
                restored_sequence.append(merge_diacritics(undiacritized_words_sequence[i],
                                                          extract_diacritics(predicted_sequence[i])))
        return restored_sequence

    @staticmethod
    def generate_words_patterns_bigrams(diacritized_word_sequences):

        def extract_words_generate_patterns(d_words_sequence):
            d_patterns_bigrams = []
            for d_w1, d_w2 in zip(d_words_sequence[:-1], d_words_sequence[1:]):
                d_patterns_bigrams.append((convert_to_pattern(d_w1), convert_to_pattern(d_w2)))
            return [convert_to_pattern(clear_diacritics(d_w)) for d_w in d_words_sequence], d_patterns_bigrams

        d_words = []
        futures = []
        pool_executor = PoolExecutor(MAX_PARALLEL_RUNS)
        for d_words_sequence in diacritized_word_sequences:
            d_words.extend(d_words_sequence)
            futures.append(pool_executor.submit(extract_words_generate_patterns, d_words_sequence))
        u_patterns_indexes = IndexDict({UNKNOWN: 1, NUMBER: 1, FOREIGN: 1})
        d_patterns_bigrams = []
        for f in as_completed(futures):
            u_ps, d_ps_bgs = f.result()
            u_patterns_indexes.update((u_pattern, 1) for u_pattern in u_ps)
            d_patterns_bigrams.extend(d_ps_bgs)
        pool_executor.shutdown(False)
        return d_words, u_patterns_indexes, d_patterns_bigrams

    @staticmethod
    def calculate_frequencies(diacritized_sequences, d_words, d_patterns_bigrams, d_patterns_indexes):

        def generate_d_pattern_u_pattern(d_words, diacritized_word_sequences):
            d_pattern_u_pattern_counter = defaultdict(Counter)  # (d_pattern, u_pattern): count.
            for d_word in d_words:
                d_pattern = convert_to_pattern(d_word)
                d_patterns_indexes[d_pattern] = 1  # This change will be reflected outside the function.
                d_pattern_u_pattern_counter[d_pattern][convert_to_pattern(clear_diacritics(d_word))] += 1
            patterns_start_counter = Counter([convert_to_pattern(s[0]) for s in diacritized_word_sequences])
            return d_pattern_u_pattern_counter, patterns_start_counter

        pool_executor = PoolExecutor(MAX_PARALLEL_RUNS)
        f1 = pool_executor.submit(lambda: Counter([convert_to_pattern(d_w) for d_w in d_words]))
        f2 = pool_executor.submit(lambda: Counter(d_patterns_bigrams))
        f3 = pool_executor.submit(generate_d_pattern_u_pattern, d_words, diacritized_sequences)
        wait([f1])
        d_patterns_unigrams_counter = f1.result()
        wait([f2])
        d_patterns_bigrams_counter = f2.result()
        wait([f3])
        d_pattern_u_pattern_counter, patterns_start_counter = f3.result()
        pool_executor.shutdown(False)
        return d_patterns_unigrams_counter, d_patterns_bigrams_counter, d_pattern_u_pattern_counter, \
               patterns_start_counter

    @staticmethod
    def calculate_emissions(d_pattern_u_pattern_counter, u_patterns_indexes, d_patterns_indexes, epsilon):
        emissions = np.zeros((len(d_patterns_indexes), len(u_patterns_indexes)))
        for d_pattern, u_pattern_count in d_pattern_u_pattern_counter.items():
            i = d_patterns_indexes[d_pattern]
            emissions[i, u_patterns_indexes[UNKNOWN]] = epsilon
            for u_pattern, count in u_pattern_count.items():
                emissions[i, u_patterns_indexes[u_pattern]] = count / sum(u_pattern_count.values()) - \
                                                              epsilon / len(u_pattern_count)
        emissions[d_patterns_indexes[NUMBER]] = np.zeros(emissions.shape[1])
        emissions[d_patterns_indexes[NUMBER]][u_patterns_indexes[NUMBER]] = 1
        emissions[d_patterns_indexes[FOREIGN]] = np.zeros(emissions.shape[1])
        emissions[d_patterns_indexes[FOREIGN]][u_patterns_indexes[FOREIGN]] = 1
        return emissions

    @staticmethod
    def calculate_transitions(d_patterns_indexes, d_patterns_bigrams_counter, d_patterns_unigrams_counter, epsilon):
        # Initialize with a small strictly positive number.
        transitions = np.ones((len(d_patterns_indexes), len(d_patterns_indexes))) * (epsilon / len(d_patterns_indexes))
        # Calculate the real probabilities.
        for d_pattern1, d_pattern2 in d_patterns_bigrams_counter.keys():
            i = d_patterns_indexes[d_pattern1]
            j = d_patterns_indexes[d_pattern2]
            # Add the true probability of the pair.
            transitions[i, j] += d_patterns_bigrams_counter[d_pattern1, d_pattern2] / \
                                 d_patterns_unigrams_counter[d_pattern1]
        transitions /= np.sum(transitions, axis=-1, keepdims=True)  # Normalize everything.
        return transitions

    @staticmethod
    def calculate_priors(patterns_start_counter, d_patterns_indexes, epsilon):
        priors = np.ones(len(d_patterns_indexes)) * epsilon / len(d_patterns_indexes)
        for state in patterns_start_counter.keys():
            priors[d_patterns_indexes[state]] += patterns_start_counter[state] / sum(patterns_start_counter.values())
        priors /= np.sum(priors)
        return priors


def viterbi(obs, a, b, pi):
    # https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    path = np.zeros(T, dtype=np.int)
    delta = np.zeros((nStates, T))
    phi = np.zeros((nStates, T))

    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    pool_executor = PoolExecutor(MAX_PARALLEL_RUNS)

    def update_delta_phi(s, t):
        delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
        phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])

    futures = []
    for time_step in range(1, T):
        for state in range(nStates):
            futures.append(pool_executor.submit(update_delta_phi, state, time_step))
    wait(futures)

    path[T - 1] = np.argmax(delta[:, T - 1])
    for time_step in range(T - 2, -1, -1):
        path[time_step] = phi[path[time_step + 1], time_step + 1]

    return path
