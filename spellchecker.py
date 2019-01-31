import sys
from trie_search import Trie
import split_join
import pickle
import re
import numpy as np
import unicodedata

ITER_CAP = 10

class LanguageModel:

    def __init__(self, FILE_PATH):
        self.dict = {}
        self.__count_of_word = 0.
        self._regex = re.compile(r"\w+")
        self.__fit(FILE_PATH)
        self.__normalize()


    def __fit(self, file):

        with open(file) as f:
            content = f.readlines()

        for line in content:

            line = line.lower()
            line = line[:-1]
            index = line.find('\t')
            if index > 0:
                line = line[index+1:]

            words = re.findall(r"(?u)\w+", line)
            for i in range(len(words)):
                word = words[i]

                if word in self.dict.keys():
                    self.dict[word]["freq"] += 1.
                else:
                    self.dict[word] = {"freq": 1.,
                                       "words": {}}

                self.__count_of_word += 1

                if i != len(words)-1:
                    if words[i+1] in self.dict[word]["words"].keys():
                        self.dict[word]["words"][words[i+1]] += 1.
                    else:
                        self.dict[word]["words"][words[i+1]] = 1.


    def __normalize(self):
        for key, value in zip(self.dict.keys(), self.dict.values()):
            value["freq"] /= self.__count_of_word

            count_of_uses = sum(value["words"].values())

            for word, freq in zip(value["words"].keys(), value["words"].values()):
                value["words"][word] /= count_of_uses


    def __get_word_prob(self, w1, w2):
        try:
            w2_prob = 1e-8
            if w2 in self.dict[w1]["words"].keys():
                w2_prob = self.dict[w1]["words"][w2]
            return self.dict[w1]["freq"] * w2_prob

        except Exception:
            return 1e-28


    def get_prob(self, query):
        if len(query) == 0:
            return 0.
        if len(query) == 1 and query[-1] not in self.dict.keys():
            return 0.

        prob = 1.
        for i in range(len(query)-1):
            w1 = query[i]
            w2 = query[i+1]
            prob *= self.__get_word_prob(w1, w2)

        if query[-1] in self.dict.keys():
            prob *= self.dict[query[-1]]["freq"]

        return prob*len(query)

    def get_word_prob(self, word):
        if word in self.dict.keys():
            return self.dict[word]["freq"]
        else:
            return 0.

def generate_features(language_model, query_inc, words_inc, query_corr, words_corr):
    en_ = re.compile(r'[a-z]')
    x = []
    max_prob = -2
    min_prob = 2
    count_of_words_in_dict = 0

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
    x.append(lm.get_prob(list(union)))

    for word in list(union):
        prob = lm.get_word_prob(word)
        if prob > max_prob:
            max_prob = prob
        if prob < min_prob:
            min_prob = prob

        if word in lm.dict.keys():
            count_of_words_in_dict += 1

    x.append(max_prob)
    x.append(min_prob)
    x.append(len(list(union))-count_of_words_in_dict)

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

    return x

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
        #self.lm = LanguageModel('queries_all.txt')
        self.trie = Trie()
        #with open('./classificator', 'r') as f:
        #    self.dicty_classifier = pickle.load(f)
        for w in self.language_model:
            self.trie.push(w)
        self.trie.set_search(self.language_model, self.error_model, self.alpha, self.threshold, self.N)

    def __call__(self, q):
        punctuation = [unichr(i) for i in range(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P')]
        punctuation = u''.join(punctuation)
        for _ in range(ITER_CAP):
            splits, split_scores = split_join.split(q, self.cond_prob, self.language_model) # splits
            csplits = []
            csplits_scores = []
            for i, fix in enumerate(splits):
                if split_join.classification(fix, q, self.cond_prob, self.language_model):
                    csplits.append(fix)
                    csplits_scores.append(split_scores[i])
            joins, join_scores = split_join.join(q, self.cond_prob, self.language_model) # join
            cjoins = []
            cjoins_scores = []
            for i, fix in enumerate(joins):
                if split_join.classification(fix, q, self.cond_prob, self.language_model):
                    cjoins.append(fix)
                    cjoins_scores.append(join_scores[i])
            wds = q.split(' ')
            query_graph = []
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
                #query_fix, words_fix = get_formated_text(fix)
                #query_query, words_query = get_formated_text(q)
                #features = generate_features(self.lm, query_query, words_query, query_fix, words_fix)
                #if self.dicty_classifier.predict(features):
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
        print(engine(query.decode('utf-8')))
    else:
        break
