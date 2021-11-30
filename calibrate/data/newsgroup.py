import os
import os.path as osp
import sys
import numpy as np
import torch
import logging

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)


def ng_loader(
    data_dir: str,
    max_sequence_length: int = 1000,
    max_num_words: int = 2000,
    embedding_dim: int = 100,
    test_split: float = 0.2,
    shuffle=True,
    random_seed=1,
):
    # BASE_DIR = 'NewsGroup'
    # GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    # TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
    # MAX_SEQUENCE_LENGTH = 1000
    # MAX_NUM_WORDS = 20000
    # EMBEDDING_DIM = 100
    # VALIDATION_SPLIT = 0.2
    logger.info("Start process 20 newsgroups text data ...")
    glove_dir = osp.join(data_dir, "glove.6B")
    text_data_dir = osp.join(data_dir, "20_newsgroups")

    logger.debug('Indexing word vectors.')

    embeddings_index = {}
    with open(
        osp.join(glove_dir, "glove.6B.{}d.txt".format(embedding_dim))
    ) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    logger.debug('Found {} word vectors.'.format(len(embeddings_index)))

    # second, prepare text samples and their labels
    logger.debug('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = osp.join(text_data_dir, name)
        if osp.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = osp.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    logger.debug('Found {} texts.'.format(len(texts)))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    logger.debug('Found {} unique tokens.'.format(len(word_index)))

    data = pad_sequences(sequences, maxlen=max_sequence_length)

    labels = to_categorical(np.asarray(labels))

    logger.info('Shape of data tensor: {}'.format(data.shape))
    logger.info('Shape of label tensor: {}'.format(labels.shape))

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_test_samples = int(test_split * data.shape[0])

    x_train = data[:-(num_test_samples + 900)]  # Train set
    y_train = labels[:-(num_test_samples + 900)]
    x_val = data[
        (data.shape[0] - num_test_samples - 900):(data.shape[0] - num_test_samples)
    ]  # Validation set
    y_val = labels[
        data.shape[0]-(num_test_samples+900):(data.shape[0]-num_test_samples)
    ]
    x_test = data[-num_test_samples:]  # Test set
    y_test = labels[-num_test_samples:]

    logger.info(data.shape[0] - num_test_samples)
    # logger.info('VAL: ', x_val.shape, data.shape)

    # logger.info('Preparing embedding matrix.', x_train.shape)

    # prepare embedding matrix
    num_words = min(max_num_words, len(word_index))
    embedding_matrix = torch.zeros(num_words, embedding_dim)
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = torch.from_numpy(embedding_vector)

    logger.info("Done with text processing")

    return embedding_matrix, x_train, y_train, x_val, y_val, x_test, y_test, num_words, embedding_dim
