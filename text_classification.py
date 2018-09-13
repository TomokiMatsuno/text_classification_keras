import tensorflow as tf
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, confusion_matrix
from paths import path2tweets, path2texts
from config import lstm_dim, mlp_dim, word_dim, pdrop_lstm, pdrop_mlp, pdrop_embs, add_bos_eos, validate_every, decay
from tools import Vocab, wakati

from keras.layers import Input, Embedding, LSTM, K
from keras.models import Model
from keras import backend as K

import numpy as np


df_tweets = pd.read_csv(path2tweets, sep='\t', header=None)
df_texts = pd.read_csv(path2texts, sep='\t', header=None)


labels = df_texts[0].tolist()
vocab = Vocab([wakati(e) for e in df_texts[1].tolist()])
tweets = [[vocab.BOS] * add_bos_eos + vocab.sent2ids(t) + [vocab.EOS] * add_bos_eos if t else [vocab.UNK] for t in df_texts[1]]


print(tf.__version__)

# train_labels, test_labels, train_data, test_data = train_test_split(df_tweets[0].tolist(), [[int(e) for e in r.strip().split()] if isinstance(r, str) else [0] for r in df_tweets[1].tolist()], test_size=0.2)
train_labels, test_labels, train_data, test_data = train_test_split(labels, tweets, test_size=0.2, random_state=0)


train_labels_data = [(l, d) for l, d in zip(train_labels, train_data)]
train_labels_data = sorted(train_labels_data, key=lambda x:len(x[1]))

train_data_splits = [[]]
kinds = 0
prev_len = 0
for tld in train_labels_data:
    if kinds > 3 or len(train_data_splits[-1]) >= 500:
            train_data_splits.append([])
            kinds = 0
    train_data_splits[-1].append(tld)
    if len(tld[1]) != prev_len:
        kinds += 1

    prev_len = len(tld[1])

lengths = [len(tld[-1][1]) for tld in train_data_splits] + [140]
dict_batch_size = dict()
idx = 0

for l in lengths:
    while idx <= l:
        dict_batch_size[idx] = l
        idx += 1


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

train_labels = [[e[0] for e in s] for s in train_data_splits]
train_data = [[e[1] for e in s]for s in train_data_splits]

[(min([len(e) for e in s]), max([len(e) for e in s])) for s in train_data]
print(train_data[0])

print(len(train_data[0]), len(train_data[1]))


print(len(train_data[0]), len(train_data[1]))

print(train_data[0])


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = len(vocab.x2i) + 4


main_input = Input(shape=(None,), dtype='int32', name='main_input')
x = Embedding(vocab_size, word_dim, input_length=None)(main_input)
l2r_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True,
                dropout=pdrop_lstm, recurrent_dropout=pdrop_lstm)
r2l_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True, go_backwards=True,
                dropout=pdrop_lstm, recurrent_dropout=pdrop_lstm)
l2r_outs, r2l_outs = l2r_lstm(x), r2l_lstm(x)
l2r_last_state, r2l_last_state = l2r_outs[1], r2l_outs[1]
keys = keras.layers.concatenate([l2r_last_state, r2l_last_state])
vals = keras.layers.concatenate([l2r_outs[0], K.reverse(r2l_outs[0], 1)])
keys = keras.layers.Dense(lstm_dim*2)(keys)

def tf_funcs(keys):
    keys = K.tf.expand_dims(keys, 2)
    attn = K.tf.nn.softmax(K.tf.matmul(vals, keys))
    tweet_vec = K.tf.reduce_sum(K.tf.multiply(attn, vals), 1)
    return tweet_vec

tweet_vec = keras.layers.Lambda(lambda inputs: tf_funcs(inputs))(keys)

score = keras.layers.Dense(1, activation='relu')(tweet_vec)
model = Model(inputs=[main_input], outputs=[score])
model.summary()
model.compile(optimizer=keras.optimizers.adam(decay=decay),
              loss='binary_crossentropy',
              metrics=["accuracy"])


x_val = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=140)


partial_x_train = train_data[2000:]

y_val = test_labels
partial_y_train = train_labels[2000:]
class_weight = {0:sum(test_labels)/len(test_labels), 1:1.}

buckets = [(len(max(td, key=len)), set([len(t) for t in td])) for td in train_data]


partial_x_train = []
for td in train_data:
    partial_x_train.append(keras.preprocessing.sequence.pad_sequences(td,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=len(max(td,key=len))))

epochs=100
for epc in range(epochs):
    steps = np.random.permutation([step for step in range(len(train_data_splits))])
    for step in steps:
        # partial_x_train = train_data[step]
        partial_y_train = train_labels[step]
        history = model.fit(partial_x_train[step],
                            partial_y_train,
                            epochs=1,
                            batch_size=None,
                            verbose=0,
                            class_weight=class_weight)
        # validation_data = (x_val, y_val),
    if epc % validate_every == 0 and epc > 0:
        failed = 0
        pred_scores = []

        for d in test_data:
            try:
                # pred_scores.append(model.predict(np.matrix(d if d else [0])))
                pred_scores.append(model.predict(np.matrix(d + [vocab.PAD] * (dict_batch_size[len(d)] - len(d)) if d else [0])))
            except:
                failed += 1
                print(failed)

        pred_labels = [int(v > 0.5) for v in pred_scores]
        print(classification_report(test_labels, pred_labels))
        print(confusion_matrix(test_labels, pred_labels))


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


