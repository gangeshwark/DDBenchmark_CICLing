import csv
from pprint import pprint

import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

import os
import pickle


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def get_deceptive():
    all_ = []
    X_test = {}
    labels = {}
    with open(
            "/projects/Datasets/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if (row['class'] == 'deceptive'):
                labels[row['id'].rsplit('.', 1)[0]] = int(1)
            else:
                labels[row['id'].rsplit('.', 1)[0]] = int(0)
    print(labels)
    y = []
    path = '/projects/Datasets/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Transcription/Deceptive/'
    for x in range(1, 62):
        file = 'trial_lie_%03d.txt' % x
        text_path = os.path.join(path, file)
        # print(text_path)
        assert os.path.isfile(text_path) == True
        data = open(text_path, 'r').readlines()
        t = ''
        for line in data:
            # print i, x
            text = line.strip()
            text = clean_str(text)
            t += text
        txt = t.split(' ')
        # print((text))
        all_.append((file[:-4], txt))
        X_test[file[:-4]] = txt
        y.append([0, 1])

    path = '/projects/Datasets/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Transcription/Truthful'
    for x in range(1, 61):
        file = 'trial_truth_%03d.txt' % x
        text_path = os.path.join(path, file)
        # print(text_path)
        assert os.path.isfile(text_path) == True
        data = open(text_path, 'r').readlines()
        t = ''
        for line in data:
            # print i, x
            text = line.strip()
            text = clean_str(text)
            t += text
        txt = t.split(' ')
        # print((text))
        all_.append((file[:-4], txt))
        X_test[file[:-4]] = txt
        y.append([1, 0])
    y = np.array(y)

    pprint(X_test)
    pprint(len(X_test.keys()))
    keys = list(sorted(X_test.keys()))
    x_text = []
    idx = {}
    for i, k in enumerate(keys):
        idx[i] = k
        # print(X_test[k])
        x_text.append(X_test[k])
    print(idx)
    print(x_text)
    with open('deception_feat_index.pkl', 'wb') as f:
        pickle.dump(idx, f)
    return x_text, y


# get_deceptive()


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    # sequence_length = 246
    print(sequence_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    print(padded_sentences)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # return
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()] + ['<UNK>']
    f = open('vocab', 'w')
    f.writelines('\n'.join(vocabulary_inv))

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    print("Vocab len:", len(vocabulary.keys()))
    print(vocabulary, vocabulary_inv)
    with open('vocabulary_inv.pkl', 'wb') as f:
        pickle.dump(vocabulary_inv, f)
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    return [vocabulary, vocabulary_inv]


def load_vocab():
    with open('vocabulary_inv.pkl', 'rb') as f:
        with open('vocabulary.pkl', 'rb') as f1:
            vocabulary_inv = pickle.load(f)
            vocabulary = pickle.load(f1)
            return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    unk = []
    for sentence in sentences:
        x1 = []
        for word in sentence:
            if word in vocabulary:
                x1.append(vocabulary[word])
            else:
                x1.append(vocabulary['<UNK>'])
                unk.append(word)
        x.append(x1)
    x = np.array(x)
    print("UNK: ", len(unk))
    y = np.array(labels)
    return x, y


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = get_deceptive()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # vocabulary, vocabulary_inv = load_vocab()
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# print(load_MOSI_data_and_labels())
# print(load_data())
#
x, y, vocabulary, vocabulary_inv = load_data()
print(x)
print(y)
print(x.shape, y.shape)
