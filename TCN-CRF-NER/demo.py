import os
import codecs
import numpy
import keras
import copy
import re

from keras_wc_embd import get_dicts_generator, get_batch_input
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from model import build_model
import tensorflow as tf

# MODEL_PATH = 'model/model.h5'
MODEL_PATH = 'model/bco.h5'
# MODEL_PATH = 'model/BCO/bco.h5'

DATA_ROOT = 'dataset/new_conll'  # dataset/CoNNL2003eng
DATA_TRAIN_PATH = 'dataset/new_conll/train.txt'  # 'dataset/CoNNL2003eng/train.txt'
DATA_VALID_PATH = 'dataset/new_conll/test.txt'  # 'dataset/CoNNL2003eng/valid.txt'
DATA_TEST_PATH = 'dataset/new_conll/test.txt'  # dataset/CoNNL2003eng/test.txt'

WORD_EMBD_PATH = 'dataset/glove.6B.100d.txt'

BATCH_SIZE = 16
EPOCHS = 10

# TAGS = {
#     'O': 0,
#     'B-PER': 1,
#     'I-PER': 2,
#     'B-LOC': 3,
#     'I-LOC': 4,
#     'B-ORG': 5,
#     'I-ORG': 6,
#     'B-MISC': 7,
#     'I-MISC': 8,
# }


# TAGS = {
#     'O': 0,
#     'B-DNA': 1,
#     'I-DNA': 2,
#     'B-RNA': 3,
#     'I-RNA': 4,
#     'B-protein': 5,
#     'I-protein': 6,
#     'B-cell_type': 7,
#     'I-cell_type': 8,
#     'B-cell_line': 9,
#     'I-cell_line': 10,
# }
TAGS = {
    'O': 0,
    'B-GENE': 1,
    'I-GENE': 2,
    'E': 3,
    'S': 4
}


def load_data(path):
    sentences, taggings = [], []
    with codecs.open(path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                if not sentences or len(sentences[-1]) > 0:
                    sentences.append([])
                    taggings.append([])
                continue
            parts = line.split()
            if parts[0] != '-DOCSTART-':
                sentences[-1].append(parts[0])
                taggings[-1].append(TAGS[parts[-1]])
    if not sentences[-1]:
        sentences.pop()
        taggings.pop()
    return sentences, taggings


print('Loading...')
train_sentences, train_taggings = load_data(DATA_TRAIN_PATH)
valid_sentences, valid_taggings = load_data(DATA_VALID_PATH)

dicts_generator = get_dicts_generator(
    word_min_freq=2,
    char_min_freq=2,
    word_ignore_case=True,
    char_ignore_case=False
)
for sentence in train_sentences:
    dicts_generator(sentence)
word_dict, char_dict, max_word_len = dicts_generator(return_dict=True)
word_dict['<EOS>'] = len(word_dict)
reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))

if os.path.exists(WORD_EMBD_PATH):
    print('Embedding...')
    word_embd_weights = [[0.0] * 100] + numpy.random.random((len(word_dict) - 1, 100)).tolist()
    with codecs.open(WORD_EMBD_PATH, 'r', 'utf8') as reader:
        for line_num, line in enumerate(reader):
            if (line_num + 1) % 1000 == 0:
                print('Load embedding... %d' % (line_num + 1), end='\r', flush=True)
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0].lower()
            if word in word_dict:
                word_embd_weights[word_dict[word]] = parts[1:]
    word_embd_weights = numpy.asarray(word_embd_weights)
    print('Dict size: %d  Shape of weights: %s' % (len(word_dict), str(word_embd_weights.shape)))
else:
    word_embd_weights = None
    print('Dict size: %d' % len(word_dict))

train_steps = (len(train_sentences) + BATCH_SIZE - 1) // BATCH_SIZE
valid_steps = (len(valid_sentences) + BATCH_SIZE - 1) // BATCH_SIZE


def batch_generator(sentences, taggings, steps, training=True):
    global word_dict, char_dict, max_word_len
    while True:
        for i in range(steps):
            batch_sentences = sentences[BATCH_SIZE * i:min(BATCH_SIZE * (i + 1), len(sentences))]
            output_sentences = copy.deepcopy(batch_sentences)
            batch_taggings = taggings[BATCH_SIZE * i:min(BATCH_SIZE * (i + 1), len(taggings))]
            word_input, char_input = get_batch_input(
                batch_sentences,
                max_word_len,
                word_dict,
                char_dict,
                word_ignore_case=True,
                char_ignore_case=False
            )
            if not training:
                # yield [word_input, char_input], batch_taggings  # original
                yield [word_input, char_input], batch_taggings, output_sentences  # 加一个原始text输出
                # yield [word_input], batch_taggings, output_sentences  # 去掉字符级
                continue
                pass
            sentence_len = word_input.shape[1]
            for j in range(len(batch_taggings)):
                batch_taggings[j] = batch_taggings[j] + [0] * (sentence_len - len(batch_taggings[j]))
                batch_taggings[j] = [[tag] for tag in batch_taggings[j]]
            batch_taggings = numpy.asarray(batch_taggings)

            yield [word_input, char_input], batch_taggings
            # yield [word_input, char_input], batch_taggings, output_sentences
        if not training:
            break


def train(_):
    model = build_model(word_dict_len=len(word_dict),
                        char_dict_len=len(char_dict),
                        max_word_len=max_word_len,
                        word_dim=100,
                        char_dim=80,
                        char_embd_dim=25,
                        output_dim=len(TAGS),
                        word_embd_weights=word_embd_weights)

    # model.load_weights('model/weights-15-0.980541.hdf5', by_name=True)
    print('Fitting...')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0)
    save_model = ModelCheckpoint('model/weights-{epoch:02d}-{val_acc:.6f}.hdf5', monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(
        generator=batch_generator(train_sentences, train_taggings, train_steps),
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=batch_generator(valid_sentences, valid_taggings, valid_steps),
        validation_steps=valid_steps,

        callbacks=[
            save_model, reduce_lr
        ],
        verbose=2,
    )

    model.save_weights(MODEL_PATH)


def test(_):
    model = build_model(word_dict_len=len(word_dict),
                        char_dict_len=len(char_dict),
                        max_word_len=max_word_len,
                        word_dim=100,
                        char_dim=80,
                        char_embd_dim=25,
                        output_dim=len(TAGS),
                        word_embd_weights=word_embd_weights)

    # model.load_weights('model/weights-15-0.958716.hdf5', by_name=True)
    model.load_weights('model/weights-10-0.948524.hdf5', by_name=True)
    test_sentences, test_taggings = load_data(DATA_TEST_PATH)
    test_steps = (len(valid_sentences) + BATCH_SIZE - 1) // BATCH_SIZE

    print('Predicting...')

    def get_tags(tags):
        filtered = []
        for i in range(len(tags)):
            if tags[i] == 0:
                continue
            if tags[i] % 2 == 1:
                filtered.append({
                    'begin': i,
                    'end': i,
                    'type': i,
                })
            elif i > 0 and tags[i - 1] == tags[i] - 1:
                filtered[-1]['end'] += 1
        return filtered

    eps = 1e-6
    total_pred, total_true, matched_num = 0, 0, 0.0
    end_text = list()  #
    end_label = list()  #
    end_pre_label = list()  #
    for inputs, batch_taggings, outputs in batch_generator(
            test_sentences,
            test_taggings,
            test_steps,
            training=False):
        predict = model.predict_on_batch(inputs)
        predict = numpy.argmax(predict, axis=2).tolist()

        end_text += outputs
        end_label += batch_taggings
        end_pre_label += predict

        for i, pred in enumerate(predict):
            pred = get_tags(pred[:len(batch_taggings[i])])
            true = get_tags(batch_taggings[i])
            total_pred += len(pred)
            total_true += len(true)
            matched_num += sum([1 for tag in pred if tag in true])
    precision = (matched_num + eps) / (total_pred + eps)
    recall = (matched_num + eps) / (total_true + eps)
    f1 = 2 * precision * recall / (precision + recall)
    print('P: %.4f  R: %.4f  F: %.4f' % (precision, recall, f1))
    with open(r'./result.txt', 'w', encoding='utf-8') as f:
        for text, label, pre_label in zip(end_text, end_label, end_pre_label):
            f.writelines(' '.join(text))
            f.write('\n')
            f.writelines(re.sub(r'\]|\[', '', str(label)))
            f.write('\n')
            f.writelines(re.sub(r'\]|\[', '', str(pre_label)))
            f.write('\n')
    print('finish')


if __name__ == "__main__":
    tf.app.run(test)  # train
