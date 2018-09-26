import tensorflow as tf
import keras
from keras.layers import Input, Embedding, LSTM
from keras.models import Model
from keras.layers.core import K

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import os

from paths import path2texts, path2parse, path2save, path2params, path2outfile
from config import lstm_dim, mlp_dim, word_dim, pdrop_lstm, pdrop_mlp, pdrop_embs, \
    add_bos_eos, validate_every, decay, all_for_training, parse, validation, epochs
from tools import Vocab, wakati, prepare_pretrained_embs

import numpy as np

df_texts = pd.read_csv(path2texts, sep='\t', header=None)
vocab = Vocab([wakati(e) for e in df_texts[1].tolist()])

if not parse:
    labels = df_texts[0].tolist()
    tweets = [[vocab.BOS] + vocab.sent2ids(t) + [vocab.EOS] if t else [vocab.UNK] for t in df_texts[1]]
else:
    df_parse = pd.read_csv(path2parse, sep='\t', header=None, engine='python').dropna()
    labels = [0 for _ in range(len(df_parse[0]))]
    tweets = [[vocab.BOS] + vocab.sent2ids(t) + [vocab.EOS] if t else [vocab.UNK] for t in df_parse[1]]

print(tf.__version__)

assert sum([validation, all_for_training]) == 1, 'You must select one of the following options: validation, all_for_training, parse.'

if validation:
    # Split a file into training data and validation data.
    train_labels, test_labels, train_data, test_data = train_test_split(labels, tweets, test_size=0.2, random_state=0)

if all_for_training:
    # Use the whole input for training.
    train_labels, test_labels, train_data, test_data = train_test_split(labels, tweets, test_size=0.01, random_state=0)
    train_labels, train_data = labels, tweets

# Prepare buckets
train_labels_data = [(l, d) for l, d in zip(train_labels, train_data)]
train_labels_data = sorted(train_labels_data, key=lambda x:len(x[1]))

train_data_splits = [[]]
kinds = 0
prev_len = 0
for tld in train_labels_data:
    # Add a new bucket if the present one has sentences of three different sizes or 500 sentences.
    if kinds > 3 or len(train_data_splits[-1]) >= 500:
            train_data_splits.append([])
            kinds = 0
    train_data_splits[-1].append(tld)
    if len(tld[1]) != prev_len:
        kinds += 1

    prev_len = len(tld[1])

# Lengths of the longest sentence in each batch.
lengths = [len(tld[-1][1]) for tld in train_data_splits] + [140]

# Dictionary from lengths of sentences to the size of the batch they should belong to.
dict_batch_size = dict()
idx = 0

for l in lengths:
    while idx <= l:
        dict_batch_size[idx] = l
        idx += 1

train_labels = [[e[0] for e in s] for s in train_data_splits]
train_data = [[e[1] for e in s] for s in train_data_splits]
partial_x_train = []
for td in train_data:
    #MODIFIED-START
    partial_x_train.append([keras.preprocessing.sequence.pad_sequences([[e if e < vocab.text_vocab_size else vocab.UNK for e in s] for s in td], value=0, padding='post',
                                                                       maxlen=len(max(td, key=len))),
                            keras.preprocessing.sequence.pad_sequences(td, value=0, padding='post',
                                                                       maxlen=len(max(td, key=len)))])
    #MODIFIED-END

buckets = [(len(max(td, key=len)), set([len(t) for t in td])) for td in train_data]

class Classifier(object):
    def __init__(self):
        # Define the neural network model
        self.vocab = vocab
        #START
        vocab_size = len(vocab.x2i) + 1
        main_input = Input(shape=(None,), dtype='int32', name='main_input')
        pret_input = Input(shape=(None,), dtype='int32', name='pret_input')
        x1 = Embedding(vocab_size, word_dim, input_length=None)(main_input)
        x2 = Embedding(vocab_size, word_dim, weights=[prepare_pretrained_embs(vocab)], input_length=None, trainable=False)(pret_input)
        x = keras.layers.Add()([x1, x2])
        #END

        l2r_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True,
                        dropout=pdrop_lstm, recurrent_dropout=pdrop_lstm)
        r2l_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True, go_backwards=True,
                        dropout=pdrop_lstm, recurrent_dropout=pdrop_lstm)
        l2r_outs, r2l_outs = l2r_lstm(x), r2l_lstm(x)
        l2r_last_state, r2l_last_state = l2r_outs[1], r2l_outs[1]
        keys = keras.layers.concatenate([l2r_last_state, r2l_last_state])
        vals = keras.layers.concatenate([l2r_outs[0], K.reverse(r2l_outs[0], 1)])
        keys = keras.layers.Dense(lstm_dim*2)(keys)

        # Wrap the functions directly borrowed from tensorflow in this function.
        def tf_funcs(keys):
            from keras import backend as K
            keys = K.tf.expand_dims(keys, 2)
            attn = K.tf.nn.softmax(K.tf.matmul(vals, keys))
            tweet_vec = K.tf.reduce_sum(K.tf.multiply(attn, vals), 1)
            return tweet_vec

        tweet_vec = keras.layers.Lambda(lambda inputs: tf_funcs(inputs))(keys)
        tweet_vec = keras.layers.LeakyReLU()(tweet_vec)
        score = keras.layers.Dense(1, activation='sigmoid')(tweet_vec)
        self.model = Model(inputs=[main_input, pret_input], outputs=[score])
        self.model.summary()
        self.model.compile(optimizer=keras.optimizers.adam(decay=decay),
                      loss='binary_crossentropy',
                      metrics=["accuracy"])


    def load(self, path2model):
        self.model.load_weights(path2model)

    # def train(self, train_data_splits, train_data, train_labels, test_labels, epochs, path2save, show_progress=False):
    def train(self, show_progress=False):
        if os.path.exists(path2save):
            print('The save directory already exists. Enter Y to continue, Control + C to abort.')
            while input() != 'Y':
                pass
        else:
            os.mkdir(path2save)

        # Calculate class weights to avoid overfitting due to its imbalanced data.
        class_weight = {0: sum([sum(tl) for tl in train_labels]) / sum([len(tl) for tl in train_labels]), 1: 1.}

        for epc in range(epochs):
            steps = np.random.permutation([step for step in range(len(train_data_splits))])
            for step in steps:
                if parse:
                    break
                partial_y_train = train_labels[step]
                history = self.model.fit(partial_x_train[step],
                                    partial_y_train,
                                    epochs=1,
                                    batch_size=None,
                                    verbose=int(show_progress),
                                    class_weight=class_weight)
                # validation_data = (x_val, y_val),
            if (epc % validate_every == 0 and epc > 0):
                pred_scores = []

                for d in test_data:
                    #START
                    val_input = d + [vocab.PAD] * (dict_batch_size[len(d)] - len(d)) if d else [0]
                    pred_scores.append(self.model.predict(
                        [np.matrix([e if e < vocab.text_vocab_size else vocab.UNK for e in val_input], dtype='int32'), np.matrix(val_input, dtype='int32')]))
                    #END
                pred_labels = [int(v > 0.5) for v in pred_scores]
                print(classification_report(test_labels, pred_labels))
                print(confusion_matrix(test_labels, pred_labels))

                self.model.save(path2save + '/parameters')

    def parse(self, tweet):

        d = [self.vocab.BOS] + self.vocab.sent2ids(tweet) + [self.vocab.EOS]
        val_input = d + [self.vocab.PAD] * (dict_batch_size[len(d)] - len(d)) if d else [0]
        val_input = [np.matrix([e if e < vocab.text_vocab_size else vocab.UNK for e in val_input], dtype='int32'), np.matrix(val_input, dtype='int32')]
        return self.model.predict(val_input)


