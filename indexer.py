import numpy as np
from Levenshtein import distance, editops
from fizzle import *
import string
import unicodedata
import sys
from trie_search import Trie
import pickle

queries_all = None
with open('queries_all.txt', 'r') as f:
    queries_all = f.readlines()
for i, query in enumerate(queries_all):
    queries_all[i] = query.decode('utf-8')

error_model = {'del' : {}, 'ins' : {}, 'sub' : {}, 'tr' : {}, 'total' : 0}
language_model = {}
total_words = 0
cond_prob = {}
cond_prob_total = 0
my_dict = Trie()
punctuation = [unichr(i) for i in range(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P')]
punctuation = u''.join(punctuation)
for i, query in enumerate(queries_all):
    if i % 5000 == 0 and i > 0:
        break
    query = query.lower()
    lst = query.split(u'\t')
    orig = lst[0]
    fix = orig
    if len(lst) > 1:
        fix = lst[1]
    ops, x = dl_distance(orig, fix)
    error_model['total'] += len(ops)
    for op in ops:
        try:
            error_model[op[0]][op[1]] += 1
        except:
            error_model[op[0]][op[1]] = 1

    nt_words = orig.split(' ')
    skipped = 0
    for w in nt_words:
        w = w.strip()
        w = w.strip(punctuation)
        if w == '':
            continue
        if w in language_model:
            language_model[w] += 1
        else:
            language_model[w] = 1
        total_words += 1
        my_dict.push(w)
    cond_prob_total += (len(nt_words) - 1 - skipped)
    for j in range(len(nt_words)-1):
        if cond_prob.get(nt_words[j]) == None:
            cond_prob[nt_words[j]] = {}
            cond_prob[nt_words[j]][nt_words[j+1]]  = 1
        else:
            try:
                cond_prob[nt_words[j]][nt_words[j+1]] += 1
            except:
                cond_prob[nt_words[j]][nt_words[j+1]] = 1

err_model = {'del' : [], 'ins' : [], 'sub' : [], 'tr' : []}
for a in error_model['del']:
    err_model['del'].append((a[0], error_model['del'][a] / float(error_model['total'])))
for a in error_model['ins']:
    err_model['ins'].append((a[0], error_model['ins'][a] / float(error_model['total'])))
for a, b in error_model['sub']:
    err_model['sub'].append((a, b, error_model['sub'][(a, b)] / float(error_model['total'])))
for a, b in error_model['tr']:
    err_model['tr'].append((a, b, error_model['tr'][(a, b)] / float(error_model['total'])))
error_model = err_model

for key1 in cond_prob.keys():
    for key2 in cond_prob[key1].keys():
        cond_prob[key1][key2] /= cond_prob_total

for w in language_model:
    language_model[w] = float(language_model[w]) / total_words

with open('./error_model.pkl', 'w') as f:
    pickle.dump(error_model, f)
with open('./language_model.pkl', 'w') as f:
    pickle.dump(language_model, f)
with open('./bigram_language_model.pkl', 'w') as f:
    pickle.dump(cond_prob, f)
