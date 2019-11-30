import os
import re
import sys
import zipfile

import numpy as np
import tensorflow as tf

from tensor2tensor.data_generators.problem import DatasetSplit
from tensor2tensor.data_generators.text_problems import Text2TextProblem, VocabType
from tensor2tensor.data_generators.text_encoder import TokenTextEncoder, EOS_ID, PAD_ID
from tensor2tensor.data_generators.generator_utils import maybe_download
from tensor2tensor import models
from tensor2tensor.utils import registry, trainer_lib, hparams_lib, t2t_model

from processing import convert_to_pattern, convert_non_arabic, clear_diacritics, NUMBER, FOREIGN


@registry.register_problem
class PatternsDiacritization(Text2TextProblem):

    def __init__(self, *args, **kwargs):
        super(PatternsDiacritization, self).__init__(*args, **kwargs)
        self.download_url = './ATB3_text.zip'

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        dataset_path = maybe_download(tmp_dir, os.path.basename(self.download_url), self.download_url)
        dataset_zip = zipfile.ZipFile(dataset_path)
        for file_name in dataset_zip.namelist():
            file_path = os.path.join(tmp_dir, file_name)
            if not os.path.exists(file_path):
                dataset_zip.extract(file_name, tmp_dir)
        dataset_split = 'dev' if dataset_split == DatasetSplit.EVAL else dataset_split
        with open(os.path.join(tmp_dir, self.dataset_filename()+'_'+dataset_split+'.txt'), encoding='utf-8')\
                as dataset_file:
            for sentence in dataset_file:
                diacritized_patterns_sentence = convert_non_arabic(convert_to_pattern(sentence.strip('\n')))
                undiacritized_patterns_sentence = clear_diacritics(diacritized_patterns_sentence)
                yield {
                    'inputs': undiacritized_patterns_sentence,
                    'targets': diacritized_patterns_sentence
                }

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        vocab_file_path = os.path.join(data_dir, self.vocab_filename)
        if not os.path.exists(vocab_file_path):
            vocab = set()
            for s in self.generate_text_for_vocab(data_dir, tmp_dir):
                vocab.update(s.split(' '))
            vocab = [self.oov_token, NUMBER, FOREIGN] + list(vocab)
            with open(os.path.join(data_dir, self.vocab_filename), 'w', encoding='utf-8') as vocab_file:
                for p in vocab:
                    print(p, file=vocab_file)
        return super(PatternsDiacritization, self).get_or_create_vocab(data_dir, tmp_dir, force_get)

    @property
    def vocab_type(self):
        return VocabType.TOKEN

    @property
    def oov_token(self):
        return '<UNK>'

    def dataset_filename(self):
        return 'dataset'

    @property
    def multiprocess_generate(self):
        return False

    def prepare_to_generate(self, data_dir, tmp_dir):
        pass

    @property
    def num_generate_tasks(self):
        return os.cpu_count()

    @property
    def num_training_examples(self):
        return None

    @property
    def is_generate_per_split(self):
        return True


def encode_batch(input_encoder, input_str_list):
    """List of str to features arrays, ready for inference"""
    assert isinstance(input_encoder, TokenTextEncoder)
    assert isinstance(input_str_list, list) and all(isinstance(s, str) for s in input_str_list)
    inputs = list(map(lambda s: input_encoder.encode(s) + [EOS_ID], input_str_list))
    max_length = len(max(inputs, key=len))
    inputs = np.stack(list(map(lambda s: s + [PAD_ID] * (max_length - len(s)), inputs)))
    return inputs.reshape((*inputs.shape, 1))


def decode_batch(target_encoder, indexes_batch):
    """2D array of ints to list of str"""
    assert isinstance(target_encoder, TokenTextEncoder)
    assert isinstance(indexes_batch, np.ndarray) and indexes_batch.ndim == 2
    indexes_batch = list(map(lambda idx: list(idx[idx > EOS_ID]), indexes_batch))  # EOS_ID = 1, PAD_ID = 0
    decoded_batch = list(map(lambda idx: target_encoder.decode(idx), indexes_batch))
    return decoded_batch


PROBLEM_NAME = '_'.join(map(str.lower, re.findall('[A-Z][a-z]*', PatternsDiacritization.__name__)))


def generate_data(tmp_dir, data_dir):
    assert isinstance(tmp_dir, str)
    assert isinstance(data_dir, str)
    problem = registry.problem(PROBLEM_NAME)
    assert isinstance(problem, PatternsDiacritization)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    problem.generate_data(data_dir, tmp_dir)


def train(model_name, data_dir, output_dir, hparams_set, override_hparams_file_path, train_steps, checkpoint_duration,
          max_checkpoints, early_stop_steps):
    problem = registry.problem(PROBLEM_NAME)
    assert isinstance(problem, PatternsDiacritization)
    os.makedirs(output_dir, exist_ok=True)
    hparams = hparams_lib.create_hparams(hparams_set)
    hparams = hparams_lib.create_hparams_from_json(override_hparams_file_path, hparams)
    run_config = trainer_lib.create_run_config(model_name, model_dir=output_dir, keep_checkpoint_max=max_checkpoints,
                                               save_checkpoints_secs=checkpoint_duration, gpu_mem_fraction=0.8)
    experiment = trainer_lib.create_experiment(run_config, hparams, model_name, PROBLEM_NAME, data_dir, train_steps,
                                               sys.maxsize, eval_early_stopping_steps=early_stop_steps,
                                               eval_early_stopping_metric='loss')
    experiment.train_and_evaluate()


def predict(model_name, data_dir, output_dir, hparams_set, override_hparams_file_path, beam_size, source_file_path,
            dest_file_path):
    tfe = tf.contrib.eager
    tfe.enable_eager_execution()
    problem = registry.problem(PROBLEM_NAME)
    assert isinstance(problem, PatternsDiacritization)
    encoders = problem.feature_encoders(data_dir)
    checkpoint_path = tf.train.latest_checkpoint(output_dir)
    hparams = hparams_lib.create_hparams(hparams_set, '', data_dir, PROBLEM_NAME)
    hparams = hparams_lib.create_hparams_from_json(override_hparams_file_path, hparams)
    pattern_diacritization_model = models.model(model_name)(hparams, tf.estimator.ModeKeys.PREDICT)
    assert isinstance(pattern_diacritization_model, t2t_model.T2TModel)
    if source_file_path:
        with open(source_file_path, encoding='utf-8') as source_file:
            sentences = source_file.readlines()
    else:
        sentences = []
        s = input('>>>')
        while s:
            sentences.append(s)
            s = input('>>>')
    input_patterns_sentences = list(map(lambda s: convert_to_pattern(convert_non_arabic(s)), sentences))
    encoded_input_batch = encode_batch(encoders['inputs'], input_patterns_sentences)
    with tfe.restore_variables_on_create(checkpoint_path):
        encoded_output_batch = pattern_diacritization_model.infer({'inputs': encoded_input_batch},
                                                                  beam_size=beam_size)['outputs'].numpy()
    output_patterns_sentences = decode_batch(encoders['targets'], encoded_output_batch)
    dest_file = sys.stdout
    if dest_file_path:
        dest_file = open(dest_file_path, 'w', encoding='utf-8')
    for s in output_patterns_sentences:
        print(s, file=dest_file)
    dest_file.close()
