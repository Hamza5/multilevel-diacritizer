import matplotlib.pyplot as plt
import numpy as np

from multilevel_diacritizer.constants import (
    re, DIACRITICS, ARABIC_LETTERS, FATHA, DAMMA, KASRA, TANWEEN_DAMMA, TANWEEN_FATHA, TANWEEN_KASRA, SUKOON, SHADDA
)

TEST_FILE_PATH = 'C:/Users/Hamza Abbad/Desktop/tashkeela_test/tashkeela_test_001.txt'
PRED_FILE_PATH = 'C:/Users/Hamza Abbad/Desktop/multilevel-diacritizer_test_output.txt'

DIACRITICS_EXTRACTION_PATTERN = re.compile(r'[%s]([%s]*)' % (''.join(ARABIC_LETTERS), ''.join(DIACRITICS)))

DIACRITICS_NAMES = {FATHA: 'Fatha', DAMMA: 'Damma', KASRA: 'Kasra', TANWEEN_FATHA: 'Tanween Fath',
                    TANWEEN_DAMMA: 'Tanween Damm', TANWEEN_KASRA: 'Tanween Kasr', SUKOON: 'Sukoon',
                    SHADDA: 'Shadda'}


def extract_diacritics(text):
    return [x.group(1) for x in DIACRITICS_EXTRACTION_PATTERN.finditer(text)]


def diacritics_names(diacritics):
    return '+'.join(DIACRITICS_NAMES[d] for d in diacritics) or 'None'


confusion_dict = {}
with open(TEST_FILE_PATH, 'rt', encoding='UTF-8') as test_file:
    with open(PRED_FILE_PATH, 'rt', encoding='UTF-8') as pred_file:
        for test_line, pred_line in zip(test_file, pred_file):
            test_diacritics, pred_diacritics = extract_diacritics(test_line), extract_diacritics(pred_line)
            for t_d, p_d in zip(test_diacritics, pred_diacritics):
                try:
                    confusion_dict[t_d][p_d] += 1
                except KeyError:
                    try:
                        confusion_dict[t_d][p_d] = 1
                    except KeyError:
                        confusion_dict[t_d] = {p_d: 1}
ys = set()
for d in confusion_dict.values():
    ys = ys.union(d.keys())
for info in confusion_dict.values():
    for p_d in ys:
        if p_d not in info:
            info[p_d] = 0

confusion_matrix = np.array([[confusion_dict[x][y] for y in sorted(ys)] for x in sorted(confusion_dict.keys())])
confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=-1, keepdims=True)
fig = plt.figure(dpi=150)
plt.imshow(confusion_matrix, cmap='Blues')
plt.ylabel('Test data')
plt.xlabel('Predicted data')
ax = plt.gca()
assert isinstance(ax, plt.Axes)
ax.set_yticks(np.arange(len(confusion_dict)))
ax.set_xticks(np.arange(len(ys)))
ax.set_yticklabels([diacritics_names(x) for x in sorted(confusion_dict.keys())], fontname='Arial', fontsize=7)
ax.set_xticklabels([diacritics_names(x) for x in sorted(ys)], fontname='Arial', fontsize=7)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", va='bottom', rotation_mode="anchor")
for i in range(len(confusion_dict)):
    for j in range(len(ys)):
        ax.text(j, i, '{:.1%}'.format(confusion_matrix[i, j]), ha="center", va="center", color="slategrey", fontsize=5)

plt.tight_layout()
plt.show()
