from fizzle import *
import math
import bisect

class Node:
    def __init__(self, id=0, value='', walk = '', final=False, prev=None, next=[]):
        self.id = id
        self.value = value
        self.walk = walk
        self.final = final
        self.prev = prev
        self.next = next

    def copy(self):
        ret = Node(id=self.id, value=self.value, final=self.final, prev=self.prev, next=self.next.copy())
        return ret

class Trie:
    def __init__(self, word=None):
        root_node = Node(next=[])
        self.nodes = [root_node]
        self.ptr = 1
        if word:
            for c in word:
                new_walk = root_node.walk + c
                new_node = Node(id=self.ptr, value=c, walk=new_walk, final=False, prev=root_node.id, next=[])
                root_node.next.append(self.ptr)
                self.nodes.append(new_node)
                root_node = new_node
                self.ptr += 1
            root_node.final = True

    def push(self, word):
        root_node = self.nodes[0]
        push_trig = False
        for c in word:
            if push_trig:
                new_walk = root_node.walk + c
                new_node = Node(id=self.ptr, value=c, walk=new_walk, final=False, prev=root_node.id, next=[])
                root_node.next.append(self.ptr)
                self.nodes.append(new_node)
                root_node = new_node
                self.ptr += 1
            else:
                s_trig = True
                for branch_node_id in root_node.next:
                    branch_node = self.nodes[branch_node_id]
                    if branch_node.value == c:
                        root_node = branch_node
                        s_trig = False
                        break
                if s_trig:
                    push_trig = True
                    new_walk = root_node.walk + c
                    new_node = Node(id=self.ptr, value=c, walk=new_walk, final=False, prev=root_node.id, next=[])
                    root_node.next.append(self.ptr)
                    self.nodes.append(new_node)
                    root_node = new_node
                    self.ptr += 1
        root_node.final = True

    def check(self, word):
        root_node = self.nodes[0]
        for c in word:
            s_trig = True
            for branch_node_id in root_node.next:
                branch_node = self.nodes[branch_node_id]
                if branch_node.value == c:
                    root_node = branch_node
                    s_trig = False
                    break
            if s_trig:
                return False
        return root_node.final

    def set_search(self, language_model, error_model, alpha, threshold, N):
        self.language_model = language_model
        def lev_distance(s1, s2):
            return dl_distance(s1, s2, substitutions=error_model['sub'],
                                       deletions=error_model['del'],
                                       insertions=error_model['ins'],
                                       transpositions=error_model['tr'])
        self.lev_distance = lev_distance
        self.alpha = alpha
        self.threshold = threshold
        self.N = N

    def reverse_walk(self, id):
        ret = ''
        if id >= self.ptr or id <= 0:
            return ret
        node = self.nodes[id]
        while node.id > 0:
            ret = node.value + ret
            node = self.nodes[node.prev]
        return ret

    def search(self, word):
        candidates = []
        weights = []
        root_node = self.nodes[0]
        nds = [(id, 1) for id in root_node.next]
        while nds:
            next_nds = []
            for nd in nds:
                prefix = self.nodes[nd[0]].walk
                thr_check = pow(self.alpha, -self.lev_distance(prefix, word[:nd[1]])[1])
                if thr_check < self.threshold:
                    continue
                if self.nodes[nd[0]].final:
                    error_coef = math.log(pow(self.alpha, -self.lev_distance(prefix, word)[1]), 2.)
                    lang_coef = self.alpha*math.log(self.language_model[prefix], 2.)
                    weight = lang_coef + error_coef
                    index = bisect.bisect_left(weights, weight)
                    candidates.insert(i, prefix)
                    weights.insert(i, weight)
                    if len(weights) > self.N:
                        weights = weight[1:]
                        candidates = candidates[1:]
                if nd[1] < len(word):
                    next_nds.append((nd[0], nd[1] + 1))
                for next_nd in self.nodes[nd[0]].next:
                    next_nds.append((next_nd, nd[1]))
                    if nd[1] < len(word):
                        next_nds.append((next_nd, nd[1] + 1))
            nds = next_nds
        return candidates[::-1], weights[::-1]

    def print_nodes(self):
        for node in self.nodes:
            print('id: %d, value: %s, prev: %s, next: %s, final: %s' % (node.id, node.value, str(node.prev), str(node.next), str(node.final)))
