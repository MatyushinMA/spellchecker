from __future__ import division
import numpy as np
import sys


def join(query, cond_prob, language_model):
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

    return fix_join, score


def split(query, cond_prob, language_model):
    fix_split = []
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

    return fix_split, score


def classification(fix_words, score, orig, cond_prob, language_model):
    orig_words = orig.split(' ')
    orig_score = 0 

    for j in range(len(orig_words)-1):
        try:
            orig_score += cond_prob[orig_words[j]][orig_words[j+1]]/language_model[orig_words[j]]
        except:
            pass
    try:
        orig_score += language_model[orig_words[j+1]]
    except:
        pass

    if orig_score >= np.maximum(score):
        return orig
        
    return fix_words[np.argmax(score)]
   