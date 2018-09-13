import pandas as pd
import MeCab
import re
import numpy as np
import config, paths
from config import min_ocr
from collections import Counter

from gensim.models import KeyedVectors

m = MeCab.Tagger('-Owakati')
# wv = KeyedVectors.load_word2vec_format(paths.path2wv, limit=100000)
wv = KeyedVectors.load_word2vec_format(paths.path2wv, limit=1000)

class Dict:
    def __init__(self, sents, initial_entries=None):
        self.cnt = Counter(sents)
        self.i2x = {}
        self.x2i = {}
        self.appeared_x2i = {}
        self.appeared_i2x = {}
        self.freezed = False
        self.initial_entries = initial_entries

        if initial_entries is not None:
            for ent in initial_entries:
                self.add_entry(ent)

        for ent in sents:
            self.add_entry(ent)

        self.freeze()

    def get_dicts(self):
        return self.i2x, self.x2i

    def add_entry(self, ent, min_ocr=min_ocr):
        if (ent not in self.x2i and self.cnt[ent] >= min_ocr) or ent in self.initial_entries:
            if not self.freezed:
                self.x2i[ent] = len(self.x2i)
            else:
                self.x2i[ent] = self.x2i['<UNK>']
            self.i2x[len(self.i2x)] = ent
        if ent in self.x2i and self.i2x[self.x2i[ent]] != ent:
            print()

    def add_entries(self, seq=None, minimal_count=0):
        if not self.freezed:
            for elem in seq:
                if self.cnt[elem] >= minimal_count and elem not in self.i2x:
                    self.i2x.append(elem)
            self.words_in_train = set(self.i2x)
        else:
            for ent in seq:
                if ent not in self.x2i:
                    self.x2i[ent] = self.x2i['<UNK>']


    def freeze(self):
        self.freezed = True

PATH2DATA = '/Users/tomoki/NLP_data/conll2018/task1/all/portuguese-dev'
PATH2TRAIN = ''
PATH2DEV = ''


class Vocab(object):
    def __init__(self, data):

        words = []
        for s in data:
            words.extend(s)

        d = Dict(words, initial_entries=['<PAD>', '<BOS>', '<EOS>', '<UNK>'])
        self.i2x, self.x2i = d.get_dicts()
        self.PAD, self.BOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.pret_embs = np.zeros((len(self.i2x) + 1, config.word_dim))

    def add_parsefile(self, data):
        chars = []
        for s in data:
            chars.extend(s)

        self._char_dict.add_entries(chars)

    def prepare_pretrained_embs(self):
        zero_vector = np.zeros(wv.vector_size)
        for i, c in enumerate(self.x2i):
            self.pret_embs[self.x2i[c]] = wv[c] if c in wv else zero_vector

        self.pret_embs = self.pret_embs / np.std(self.pret_embs)
        return

    def sent2ids(self, t):
        t = omit_url_and_id(t)
        words = m.parse(t).strip().split()
        ret = []
        for w in words:
            ret.append(self.x2i[w] if w in self.x2i else self.x2i['<UNK>'])
        return ret

def omit_url_and_id(s):
    ptn_url, ptn_id = re.compile('http.*[a-zA-Z0-9_\/:]??'), re.compile('@[a-zA-Z0-9_]+')
    s_tmp = re.sub(ptn_url, '', s)
    ret = re.sub(ptn_id, '', s_tmp)

    return ret

def wakati(s):
    tmp = omit_url_and_id(s)
    words = m.parse(tmp).strip().split()
    return words


def preprocess_tweets(path_in, path_out, sep=' ', end=' ', add_id=False, neologd=False):
    mcb = MeCab.Tagger('-Owakati' + (' -d /Users/tomoki/macports/lib/mecab/dic/mecab-ipadic-neologd' if neologd else ''))
    df = pd.read_csv(path_in, '\t', header=None, skiprows=0)
    tweets = []

    if add_id:
        for id, line in zip(df[0], df[1]):
            tweets.append(str(id) + '\t' + mcb.parse(omit_url_and_id(line).strip()))
    else:
        for line in df[2].tolist():
            tweets.append(mcb.parse(omit_url_and_id(line).strip()).strip().split(' '))

    with open(path_out, 'w') as f:
        for t in tweets:
            if add_id:
                f.write(t)
            else:
                f.write(sep.join(t) + end)
    return

def prepare_yamesou_vocab(file):
    ret = dict()
    rank = 0

    with open(file, 'r') as f:
        for line in f.readlines():
            words = line.strip().split()
            for w in words:
                ret[w] = rank
                rank += 1

    return ret
