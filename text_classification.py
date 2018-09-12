import tensorflow as tf
# from tensorflow import keras
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, confusion_matrix
from metric_prf import P, R, F
from paths import path2tweets
from config import lstm_dim, mlp_dim, word_dim, pdrop_lstm, pdrop_mlp, pdrop_embs, add_bos_eos
from utils import batch_matmul

from keras.layers import Input, Embedding, LSTM, K
from keras.models import Model
from keras import backend as K

import numpy as np


df_tweets = pd.read_csv(path2tweets, sep='\t', header=None)

print(tf.__version__)

# imdb = keras.datasets.imdb
#
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# train_labels, test_labels, train_data, test_data = train_test_split(df_tweets[0].tolist(), [[int(e) for e in r.strip().split()] for r in df_tweets[1].tolist()], test_size=0.2)
train_labels, test_labels, train_data, test_data = train_test_split(df_tweets[0].tolist(), [[int(e) for e in r.strip().split()] if isinstance(r, str) else [0] for r in df_tweets[1].tolist()], test_size=0.2)

if add_bos_eos:
    tmp = []
    for td in train_data:
        tmp.append([0] + td)
    train_data = tmp
    tmp = []
    for td in test_data:
        tmp.append([0] + td)
    test_data = tmp


train_labels_data = [(l, d) for l, d in zip(train_labels, train_data)]
train_labels_data = sorted(train_labels_data, key=lambda x:len(x[1]))

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

train_data_splits = []
step_size = len(train_labels_data) // 64
for i in range(0, len(train_labels_data), step_size):
    train_data_splits.append(train_labels_data[i: i + step_size])


train_labels = [[e[0] for e in s] for s in train_data_splits]
train_data = [[e[1] for e in s]for s in train_data_splits]

[(min([len(e) for e in s]), max([len(e) for e in s])) for s in train_data]
print(train_data[0])

print(len(train_data[0]), len(train_data[1]))

# # A dictionary mapping words to an integer index
# word_index = imdb.get_word_index()
#
# # The first indices are reserved
# word_index = {k:(v+3) for k,v in word_index.items()}
# word_index["<PAD>"] = 0
# word_index["<START>"] = 1
# word_index["<UNK>"] = 2  # unknown
# word_index["<UNUSED>"] = 3
#
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#
# def decode_review(text):
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])
#
# decode_review(train_data[0])


# train_data_fwd = keras.preprocessing.sequence.pad_sequences(train_data,
#                                                         value=0,
#                                                         padding='pre',
#                                                         maxlen=256)

# test_data_fwd = keras.preprocessing.sequence.pad_sequences(test_data,
#                                                        value=0,
#                                                        padding='pre',
#                                                        maxlen=256)


print(len(train_data[0]), len(train_data[1]))

print(train_data[0])


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 50000


main_input = Input(shape=(None,), dtype='int32', name='main_input')
x = Embedding(vocab_size, word_dim, input_length=None)(main_input)
l2r_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True,
                dropout=pdrop_lstm, recurrent_dropout=pdrop_lstm)
r2l_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True, go_backwards=True,
                dropout=pdrop_lstm, recurrent_dropout=pdrop_lstm)
l2r_outs, r2l_outs = l2r_lstm(x), r2l_lstm(x)
l2r_last_state, r2l_last_state = l2r_outs[1], r2l_outs[1]
keys = keras.layers.concatenate([l2r_last_state, r2l_last_state])
# l2r_outs = K.tf.concat(l2r_outs, axis=1)
# r2l_outs = K.tf.concat(r2l_outs, axis=1)
# vals = K.tf.concat([l2r_outs, r2l_outs])
vals = keras.layers.concatenate([l2r_outs[0], K.reverse(r2l_outs[0], 1)])
keys = keras.layers.Dense(lstm_dim*2)(keys)

def tf_funcs(keys):
    # keys = K.tf.expand_dims(keys, 1)
    # attn = K.tf.nn.softmax(K.tf.matmul(vals, K.tf.transpose(keys,perm=[0,2,1])))
    keys = K.tf.expand_dims(keys, 2)
    attn = K.tf.nn.softmax(K.tf.matmul(vals, keys))
    # tweet_vec = K.tf.reduce_sum(K.tf.multiply(K.repeat_elements(attn, lstm_dim*2, axis=2), vals), 1)
    tweet_vec = K.tf.reduce_sum(K.tf.multiply(attn, vals), 1)
    return tweet_vec

tweet_vec = keras.layers.Lambda(lambda inputs: tf_funcs(inputs))(keys)

score = keras.layers.Dense(1, activation='relu')(tweet_vec)
model = Model(inputs=[main_input], outputs=[score])
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=["accuracy"])







# keys = keras.layers.Dense(lstm_dim * 2)
#
#
#
#
#
#
#
#
#
#
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, word_dim))
# # model.add(keras.layers.GlobalAveragePooling1D())
# # model.add(keras.layers.GlobalMaxPooling1D())
#
# lstm_outs = keras.layers.Bidirectional(keras.layers.LSTM(lstm_dim, return_sequences=True))
# keys = keras.layers.concatenate([lstm_outs[0], lstm_outs[-1]])
# lstm_outs = keras.layers.concatenate(lstm_outs, axis=1)
# bc = keras.layers.Dense(lstm_dim * 2)
#
# # model.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_dim, return_sequences=True)))
#
# model.add(keras.layers.Dense(mlp_dim, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
#
# model.summary()
#
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='binary_crossentropy',
#               metrics=["accuracy"])
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='binary_crossentropy',
#               metrics=[P, R, F])



# train_data = keras.preprocessing.sequence.pad_sequences(train_data,
#                                                         value=0,
#                                                         padding='post',
#                                                         maxlen=140)


x_val = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=140)


# x_val = train_data[:2000]
# x_val = test_data
partial_x_train = train_data[2000:]

# y_val = train_labels[:2000]
y_val = test_labels
partial_y_train = train_labels[2000:]
class_weight = {0:sum(test_labels)/len(test_labels), 1:1.}

partial_x_train = []
for td in train_data:
    partial_x_train.append(keras.preprocessing.sequence.pad_sequences(td,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=len(max(td,key=len))+1))

epochs=100
for epc in range(epochs):
    steps = np.random.permutation([step for step in range(len(train_data_splits))])
    for step in steps:
        # partial_x_train = train_data[step]
        partial_y_train = train_labels[step]
        history = model.fit(partial_x_train[step],
                            partial_y_train,
                            epochs=1,
                            batch_size=500,
                            verbose=1,
                            class_weight=class_weight)
        # validation_data = (x_val, y_val),
    if epc % 2 == 0 and epc > 0:
        failed = 0
        pred_scores = []

        for d in test_data:
            try:
                pred_scores.append(model.predict(np.matrix(d + [0] * add_bos_eos if d else [0])))
                # if add_bos_eos:
                #     pred_scores.append(model.predict(np.matrix(d + [0] if d else [0])))
                # else:
                #     pred_scores.append(model.predict(np.matrix(d if d else [0])))
            except:
                failed += 1
                print(failed)

        pred_labels = [int(v > 0.5) for v in pred_scores]
        print(classification_report(test_labels, pred_labels))
        print(confusion_matrix(test_labels, pred_labels))

# pred_labels = [model.predict(np.matrix(d)) for d in test_data]

#
# failed = 0
# pred_labels = []
#
# for d in test_data:
#     try:
#         pred_labels.append(model.predict(np.matrix(d if d else [0])))
#     except:
#         failed += 1
#         print(failed)
#
# pred_labels = [int(v > 0.5) for v in pred_labels]
# print(classification_report(test_labels, pred_labels))
# print(confusion_matrix(test_labels, pred_labels))

# print(results)

history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history.history['acc']
# val_acc = history.history['val_acc']
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


