#coding:utf-8
#adopted from here: https://gist.github.com/yano/3a072e5e2b7a55703028751820bfacbf

# import keras.backend as K
from tensorflow import keras as K

K = K.backend
# ... kerasのコードなんとかかんとか


# precision, recall, f-measureを定義する
# 0.20というのが閾値になっているので適宜変更する

#precision
def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall

#f-measure
def F(y_true, y_pred):
    p_val = P(y_true, y_pred)
    r_val = R(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)

    return f_val


# ... kerasのコードなんとかかんとか


# # metricsで学習時にP,R,Fを表示するようにする
# model.compile(optimizer=rms_prop, loss="binary_crossentropy", metrics=[P, R, F])


# ... kerasのコードなんとかかんとか