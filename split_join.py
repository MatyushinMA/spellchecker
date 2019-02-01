from __future__ import division
import numpy as np
import sys


def join(query, cond_prob, language_model):
    if query.find(' ') == -1:
        try:
            score = language_model[query]
        except:
            score = 0
        return query, score
    lst = query.split(' ')
    fix_join = []
    for i in range(len(query)):
        if query[i] == ' ':
            fix_join.append(query[:i]+query[i+1:])

    score = np.zeros(len(fix_join))
    for i in range(len(fix_join)):
        words = fix_join[i].split(' ')
        for j in range(len(words)-1):
            try:
                score[i] += cond_prob[words[j]][words[j+1]]/language_model[words[j]]
            except:
                pass

    return fix_join[np.argmax(score)], score.max()


def split(query, cond_prob, language_model):
    fix_split = []
    if len(query) == 1:
        score = language_model.get(query, 0.)
        return query, score
    for i in range(1, len(query)):
        if query[i] != ' ':
            fix_split.append(query[:i] + ' ' + query[i:])

    score = np.zeros(len(fix_split))
    for i in range(len(fix_split)):
        words = fix_split[i].split(' ')
        for j in range(len(words)-1):
            try:
                score[i] += cond_prob[words[j]][words[j+1]]/language_model[words[j]]
            except:
                pass

    return fix_split[np.argmax(score)], score.max()


def classification(fix, score, orig, cond_prob, language_model):
    orig_words = orig.split(' ')
    orig_score = 0

    for j in range(len(orig_words)-1):
        try:
            orig_score += cond_prob[orig_words[j]][orig_words[j+1]]/language_model[orig_words[j]]
        except:
            pass

    if orig_score >= score:
        return orig, orig_score
    return fix, score
