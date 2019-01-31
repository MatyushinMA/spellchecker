#-*- encoding: utf-8 -*-
'''TODO:
'''
from collections import defaultdict
import numpy as np

def dl_distance(s1,
                s2,
                substitutions=[],
                deletions=[],
                insertions=[],
                transpositions=[],
                symetric=True,
                returnMatrix=False,
                printMatrix=False,
                nonMatchingEnds=False,
                transposition=True,
                secondHalfDiscount=False,
                getops=True):
    """
	Return DL distance between s1 and s2. Default cost of substitution, insertion, deletion and transposition is 1
	substitutions is list of tuples of characters (what, substituted by what, cost), 
		maximal value of substitution is 2 (ie: cost deletion & insertion that would be otherwise used)
		eg: substitutions=[('a','e',0.4),('i','y',0.3)]
	symetric=True mean that cost of substituting A with B is same as B with A
	returnMatrix=True: the matrix of distances will be returned, if returnMatrix==False, then only distance will be returned
	printMatrix==True: matrix of distances will be printed
	transposition=True (default): cost of transposition is 1. transposition=False: cost of transpositin is Infinity
	"""
    #if not isinstance(s1, unicode):
    #    s1 = unicode(s1, "utf8")
    #if not isinstance(s2, unicode):
    #    s2 = unicode(s2, "utf8")

    subs = defaultdict(lambda: 1)  #default cost of substitution is 1
    for a, b, v in substitutions:
        subs[(a, b)] = v
        if symetric:
            subs[(b, a)] = v

    dels = defaultdict(lambda: 1)
    for a, v in deletions:
        dels[a] = v

    ins = defaultdict(lambda: 1)
    for a, v in insertions:
        ins[a] = v

    trs = defaultdict(lambda: 1)
    for a, b, v in transpositions:
        trs[(a, b)] = v
        if symetric:
            trs[(b, a)] = v

    opsmatrix = []
    matrix = []
    if nonMatchingEnds:
        matrix = [[j for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]
        if getops:
            opsmatrix = [[j for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]
    else:  #start and end are aligned
        matrix = [[i + j for j in range(len(s2) + 1)]
                  for i in range(len(s1) + 1)]
        if getops:
            opsmatrix = [[i + j for j in range(len(s2) + 1)]
                  for i in range(len(s1) + 1)]
    #matrix |s1|+1 x |s2|+1 big. Only values at border matter
    half1 = len(s1) / 2
    half2 = len(s2) / 2
    for i in range(len(s1)):
        for j in range(len(s2)):
            ch1, ch2 = s1[i], s2[j]
            if ch1 == ch2:
                cost = 0
            else:
                cost = subs[(ch1, ch2)]
            deletionCost = dels[ch1]
            insertionCost = ins[ch2]
            transpositionCost = trs[(ch1, ch2)]
            # if secondHalfDiscount and (s1 > half1 or s2 > half2):
            #     deletionCost, insertionCost = 0.6, 0.6
            # else:
            #     deletionCost, insertionCost = 1, 1

            cand = np.array([
                matrix[i][j + 1] + deletionCost,  #deletion
                matrix[i + 1][j] + insertionCost,  #insertion
                matrix[i][j] + cost  #substitution or no change
            ])
            way = np.argmin(cand)
            matrix[i + 1][j + 1] = cand[way]
            if way == 0:
                opsmatrix[i + 1][j + 1] = ('del', (ch1,))
            elif way == 1:
                opsmatrix[i + 1][j + 1] = ('ins', (ch2,))
            else:
                if cost > 0.:
                    opsmatrix[i + 1][j + 1] = ('sub', (ch1, ch2))
                else:
                    opsmatrix[i + 1][j + 1] = ('eq', (ch1, ch2))

            if transposition and i >= 1 and j >= 1 and s1[i] == s2[j -
                                                                   1] and s1[i
                                                                             -
                                                                             1] == s2[j]:
                cand = np.array([matrix[i + 1][j + 1], matrix[i - 1][j - 1] + transpositionCost])
                way = np.argmin(cand)
                matrix[i + 1][j + 1] = cand[way]
                if way == 1:
                    opsmatrix[i + 1][j + 1] = ('tr', (ch1, ch2))

    if printMatrix:
        print("     ")
        for i in s2:
            print(I, "")
        print
        for i, r in enumerate(matrix):
            if i == 0:
                print(" ", r)
            else:
                print(s1[i - 1], r)
    if printMatrix:
        print("     ")
        for i in s2:
            print(I, "")
        print()
        for i, r in enumerate(opsmatrix):
            if i == 0:
                print(" ", r)
            else:
                print(s1[i - 1], r)
    if returnMatrix:
        return matrix
    else:
        if getops:
            path = []
            pos = [len(s1), len(s2)]
            t = 0
            while pos[0] > 0 and pos[1] > 0:
                v = opsmatrix[pos[0]][pos[1]]
                if v[0] != 'eq':
                    if t == 0:
                        path.append(v)
                    elif path[-1][0] != 'tr':
                        path.append(v)
                    t += 1
                if v[0] == 'tr' or v[0] == 'eq' or v[0] == 'sub':
                    pos[0] -= 1
                    pos[1] -= 1
                elif v[0] == 'ins':
                    pos[1] -= 1
                else:
                    pos[0] -= 1
            if pos[0] == 0:
                for i in range(pos[1]):
                    path.append(('ins', (s2[i],)))
            else:
                for i in range(pos[0]):
                    path.append(('del', (s1[i],)))
            return path[::-1], matrix[-1][-1]
        return matrix[-1][-1]


def dl_ratio(s1, s2, **kw):
    'returns distance between s1&s2 as number between [0..1] where 1 is total match and 0 is no match'
    try:
        return 1 - (dl_distance(s1, s2, **kw)) / (2.0 * max(len(s1), len(s2)))
    except ZeroDivisionError:
        return 0.0


def match_list(s, l, **kw):
    '''
	returns list of elements of l with each element having assigned distance from s
	'''
    return map(lambda x: (dl_distance(s, x, **kw), x), l)


def pick_N(s, l, num=3, **kw):
    ''' picks top N strings from options best matching with s 
		- if num is set then returns top num results instead of default three
	'''
    return sorted(match_list(s, l, **kw))[:num]


def pick_one(s, l, **kw):
    try:
        return pick_N(s, l, 1, **kw)[0]
    except IndexError:
        return None


def substring_match(text, s, transposition=True,
                    **kw):  #TODO: isn't backtracking too greedy?
    """
	fuzzy substring searching for text in s
	"""
    for k in ("nonMatchingEnds", "returnMatrix"):
        if kw.has_key(k):
            del kw[k]

    matrix = dl_distance(
        s, text, returnMatrix=True, nonMatchingEnds=True, **kw)

    minimum = float('inf')
    minimumI = 0
    for i, row in enumerate(matrix):
        if row[-1] < minimum:
            minimum = row[-1]
            minimumI = i

    x = len(matrix[0]) - 1
    y = minimumI

    #backtrack:
    while x > 0:
        locmin = min(matrix[y][x - 1], matrix[y - 1][x - 1], matrix[y - 1][x])
        if matrix[y - 1][x - 1] == locmin:
            y, x = y - 1, x - 1
        elif matrix[y - 1][x] == locmin:
            y = y - 1
        elif matrix[y][x - 1] == locmin:
            x = x - 1

    return minimum, (y, minimumI)


def substring_score(s, text, **kw):
    return substring_match(s, text, **kw)[0]


def substring_position(s, text, **kw):
    return substring_match(s, text, **kw)[1]


def substring_search(s, text, **kw):
    score, (start, end) = substring_match(s, text, **kw)
    # print score, (start, end)
    return text[start:end]


def match_substrings(s, l, score=False, **kw):
    'returns list of elements of l with each element having assigned distance from s'
    return map(lambda x: (substring_score(x, s, **kw), x), l)


def pick_N_substrings(s, l, num=3, **kw):
    ''' picks top N substrings from options best matching with s 
		- if num is set then returns top num results instead of default three
	'''
    return sorted(match_substrings(s, l, **kw))[:num]


def pick_one_substring(s, l, **kw):
    try:
        return pick_N_substrings(s, l, 1, **kw)[0]
    except IndexError:
        return None


if __name__ == "__main__":
    #examples:

    commonErrors = [('a', 'e', 0.4), ('i', 'y', 0.3), ('m', 'n', 0.5)]
    misspellings = [
        "Levenshtain", "Levenstein", "Levinstein", "Levistein", "Levemshtein"
    ]

    print(dl_distance('dayme', 'dayne', substitutions=commonErrors))
    print(dl_ratio("Levenshtein", "Levenshtein"))
    print(match_list("Levenshtein", misspellings, substitutions=commonErrors))
    print(pick_N("Levenshtein", misspellings, 2, substitutions=commonErrors))
    print(pick_one("Levenshtein", misspellings, substitutions=commonErrors))

    print(substring_search(
        "aaaabcegf", "qqqqq aaaaWWbcdefg qqqq", printMatrix=True))
