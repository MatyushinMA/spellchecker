# -*- coding: utf-8 -*-

import sys
from trie_search import Trie
import split_join
import pickle
import re
import numpy as np
import unicodedata
import sklearn
from fizzle import dl_distance
from string import maketrans   # Required to call maketrans function.
import string

ITER_CAP = 10

def get_prob(language_model, cond_prob, query):
    if len(query) == 0:
        return 0.
    if len(query) == 1 and query[-1] not in language_model.keys():
        return 0.

    prob = 1.
    for i in range(len(query)-1):
        w1 = query[i]
        w2 = query[i+1]

        try:
            w2_prob = 1e-8
            if w2 in cond_prob[w1].keys():
                w2_prob = cond_prob[w1][w2] / language_model[w1]
            prob_a = language_model[w1] * w2_prob

        except Exception:
            return 1e-28

        prob *= prob_a

    if query[-1] in language_model.keys():
        prob *= language_model[query[-1]]

    return prob*len(query)

def format_text(self, words):
    formatted_query = ""
    if len(words) != len(separators):
        return " ".join(words)

    try:
        for i in range(len(words)):
            word = words[i]
            if self.init_words[i][0].isupper():
                w = word[0].upper()
                word = word[1:]
                word = w + word

            formatted_query += word
            formatted_query += separators[i].encode("utf-8")

    except Exception:
        formatted_query = " ".join(words)

    return formatted_query

def get_formated_text(text):
    if text[-1] == u"\n":
        text = text[:-1]

    init_words = re.findall(r"(?u)\w+", text)
    separators = re.split(r"(?u)\w+", text)[1:]

    text = text.lower()
    words = re.findall(r"(?u)\w+", text)

    query = text

    return query, words

def generate_features(language_model, cond_prob, query_inc, words_inc, query_corr, words_corr):
    en_ = re.compile(r'[a-z]')
    x = []
    max_prob = -2
    min_prob = 2
    count_of_words_in_dict = 0

    dl_ = dl_distance(query_inc, query_corr)

    lev = dl_[1]
    sub = 0
    ins = 0
    del_ = 0
    tr = 0

    for i in dl_[0]:
        if i[0] =='ins':
            ins = ins + 1
        elif i[0] == 'sub':
            sub = sub + 1
        elif i[0] == 'del':
            del_ = del_ + 1
        elif i[0] == 'tr':
            tr = tr + 1

    x.append(lev)
    x.append(ins)
    x.append(sub)
    x.append(del_)
    x.append(tr)

    len_words_inc = len(words_inc)
    len_query_inc= len(query_inc)
    len_words_corr= len(words_corr)
    len_query_corr= len(query_corr)

    set_words_inc = set(words_inc)
    set_words_corr = set(words_corr)

    unique = set_words_inc.symmetric_difference(set_words_corr)
    union = set_words_inc.union(set_words_corr)

    x.append(len_words_inc)
    x.append(len_query_inc)
    x.append(len_words_corr)
    x.append(len_query_corr)
    x.append(get_prob(language_model, cond_prob, list(union)))

    for word in list(union):
        # prob = lm.get_word_prob(word)
        if word in language_model.keys():
            prob = language_model[word]
        else:
            prob = 0.
        # prob = language_model[word]
        if prob > max_prob:
            max_prob = prob
        if prob < min_prob:
            min_prob = prob

        if word in language_model.keys():
            count_of_words_in_dict += 1

    x.append(max_prob)
    x.append(min_prob)
    x.append(len(list(union)) - count_of_words_in_dict)

    if u"," in query_inc or u"." in query_inc or u"'" in query_inc or u";" in query_inc or u"]" in query_inc or u"[" in query_inc or \
        u"~" in query_inc or u"," in query_corr or u"." in query_corr or u"'" in query_corr or u";" in query_corr or u"]" in query_corr or u"[" in query_corr or \
        u"~" in query_corr:
        x.append(1)
    else:
        x.append(0)

    if en_.findall(query_corr) or en_.findall(query_inc):
        lang = 1
    else:
        lang = 0
    x.append(lang)

    return np.array(x).reshape(1, -1)

class SpellChecker:
    def __init__(self, alpha=1000., threshold=0.95, N=3):
        self.alpha = alpha
        self.threshold = threshold
        self.N = N
        with open('./language_model.pkl', 'r') as f:
            self.language_model = pickle.load(f)
        with open('./error_model.pkl', 'r') as f:
            self.error_model = pickle.load(f)
        with open('./bigram_language_model.pkl', 'r') as f:
            self.cond_prob = pickle.load(f)
        self.trie = Trie()
        with open('./classificator.pkl', 'rb') as f:
            self.dicty_classifier = pickle.load(f)
        for w in self.language_model:
            self.trie.push(w)
        self.trie.set_search(self.language_model, self.error_model, self.alpha, self.threshold, self.N)
        intab = "qwertyuiop[]asdfghjkl;'\\`zxcvbnm,./йцукенгшщзхъфывапролджэ\\]ячсмитьбю.".decode('utf-8')
        outtab = "йцукенгшщзхъфывапролджэ\\]ячсмитьбю.qwertyuiop[]asdfghjkl;'\\`zxcvbnm,./".decode('utf-8')
        self.transtab = dict((ord(intab[i]), ord(outtab[i])) for i in range(len(intab)))

    def __call__(self, q):
        punctuation = [unichr(i) for i in range(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P')]
        punctuation = unicode('').join(punctuation)
        for _ in range(ITER_CAP):

            wds = q.split(unicode(' '))

            for word in wds: # layout
                layout_suggest = []
                translited_word = word.translate(self.transtab)
                try:
                    score = self.trie.search(word)[1][0]
                except:
                    score = -float('inf')
                try:
                    translited_score = self.trie.search(translited_word)[1][0]
                except:
                    translited_score = -float('inf')
                if translited_score > score:
                    layout_suggest.append(translited_word)
                else:
                    layout_suggest.append(word)
            layout_score = self.language_model.get(layout_suggest[0], 0.)
            for i, w in enumerate(layout_suggest[:-1]):
                try:
                    layout_score += self.cond_prob[w][layout_suggest[i + 1]]/self.language_model[w]
                except:
                    pass
            q = unicode(' ').join(layout_suggest)

            splits, split_scores = split_join.split(q, self.cond_prob, self.language_model) # splits
            csplits = []
            csplits_scores = []

            fix, score = split_join.classification(splits, split_scores, query, self.cond_prob, self.language_model)
            csplits.append(fix)
            csplits_scores.append(score)

            joins, join_scores = split_join.join(query, self.cond_prob, self.language_model)  # join
            cjoins = []
            cjoins_scores = []

            fix, score = split_join.classification(joins, score, query, self.cond_prob, self.language_model)
            cjoins.append(fix)
            cjoins_scores.append(score)

            wds = q.split(unicode(' '))

            query_graph = [] # dicty
            for word in wds:
                word.strip()
                word.strip(punctuation)
                dicty, dicty_scores = self.trie.search(word)
                query_graph.append(dicty)
            def recursive_fixes_search(query_graph, now_query=[], now_score=0., step=0):
                ret = []
                if step < len(query_graph) - 1:
                    for cand in query_graph[step]:
                        if not step == 0:
                            new_now_query = now_query[:] + [cand]
                            try:
                                new_now_score = now_score + self.cond_prob[now_query[-1]][cand]/self.language_model[now_query[-1]]
                            except:
                                new_now_score = now_score
                            ret += recursive_fixes_search(query_graph, new_now_query, new_now_score, step + 1)
                        else:
                            ret += recursive_fixes_search(query_graph, [cand], self.language_model[cand], step + 1)
                else:
                    for cand in query_graph[step]:
                        if now_query:
                            new_now_query = now_query + [cand]
                            try:
                                score = now_score + self.cond_prob[now_query[-1]][cand]/self.language_model[now_query[-1]]
                            except:
                                score = now_score
                            q = u' '.join(new_now_query)
                            ret.append((q, score))
                        else:
                            score = self.language_model[cand]
                            q = cand
                            ret = [(cand, score)]
                return ret
            dicty = recursive_fixes_search(query_graph)
            cdicty = []
            cdicty_scores = []
            for i, fix in enumerate(dicty):
                query_fix, words_fix = get_formated_text(fix[0])
                query_query, words_query = get_formated_text(q)
                features = generate_features(self.language_model, self.cond_prob, query_query, words_query, query_fix, words_fix)
                if self.dicty_classifier.predict(features):
                    cdicty.append(fix[0])
                    cdicty_scores.append(fix[1])
            fixes = csplits + cjoins + cdicty
            if fixes:
                scores = csplits_scores + cjoins_scores + cdicty_scores
                scores = np.array(scores)
                sort_indices = np.argsort(scores)[::-1]
                fix = fixes[sort_indices[0]]
                if q == fix:
                    return q
                q = fix
            else:
                return q


engine = SpellChecker()

while True:
    query = sys.stdin.readline().strip()
    if query:
        query = query.decode('utf-8')
        print(engine(query))
    else:
        break
