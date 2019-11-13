import os

from tensor2tensor.data_generators.problem import DatasetSplit
from tensor2tensor.data_generators.text_problems import Text2TextProblem, VocabType, txt_line_iterator
from tensor2tensor.utils import registry

from processing import convert_to_pattern, convert_non_arabic, clear_diacritics


@registry.register_problem
class PatternsDiacritization(Text2TextProblem):

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        dataset_split = 'dev' if dataset_split == DatasetSplit.EVAL else dataset_split
        for sentence in txt_line_iterator(os.path.join(tmp_dir, self.dataset_filename()+'_'+dataset_split+'_text.txt')):
            diacritized_patterns_sentence = convert_non_arabic(convert_to_pattern(sentence))
            undiacritized_patterns_sentence = clear_diacritics(diacritized_patterns_sentence)
            yield {
                'inputs': undiacritized_patterns_sentence,
                'targets': diacritized_patterns_sentence
            }

    @property
    def vocab_type(self):
        return VocabType.TOKEN

    @property
    def oov_token(self):
        return '<UNK>'

    def dataset_filename(self):
        return 'ATB3'

    @property
    def multiprocess_generate(self):
        return False

    @property
    def num_generate_tasks(self):
        return os.cpu_count()

    @property
    def num_training_examples(self):
        return None

    def prepare_to_generate(self, data_dir, tmp_dir):
        pass

    @property
    def is_generate_per_split(self):
        return True


if __name__ == '__main__':
    from tensor2tensor import problems
    from tensor2tensor.utils import trainer_lib
    problem_name = 'patterns_diacritization'
    model_name = 'transformer'
    t2t_p = problems.problem(problem_name)
    assert isinstance(t2t_p, Text2TextProblem)
    tmp_dir = './dataset/'
    data_dir = tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    vocab = {t2t_p.oov_token}
    vocab_file_path = os.path.join(data_dir, t2t_p.vocab_filename)
    if not os.path.exists(vocab_file_path):
        for s in t2t_p.generate_text_for_vocab(data_dir, tmp_dir):
            vocab.update(s.split(' '))
        with open(os.path.join(data_dir, t2t_p.vocab_filename), 'w') as vocab_file:
            for p in vocab:
                print(p, file=vocab_file)
    t2t_p.generate_data(data_dir, tmp_dir)
    hparams = trainer_lib.create_hparams('transformer_base')
    hparams.batch_size = 1024
    hparams.learning_rate_warmup_steps = 45000
    hparams.learning_rate = 0.4
    run_config = trainer_lib.create_run_config(model_name, model_dir='./',  save_checkpoints_secs=60*15)
    experiment = trainer_lib.create_experiment(run_config, hparams, model_name, problem_name, data_dir, 1000, 100)
    experiment.train_and_evaluate()
