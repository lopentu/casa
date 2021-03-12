from itertools import chain
import re

def make_ngrams(dep_data, win=4):
    words = list(chain.from_iterable(x[0] for x in dep_data))
    tok_idx = list(chain.from_iterable([idx]*len(x[0])
                for idx, x in enumerate(dep_data)))
    ngrams = []
    tok_indices = []
    for cur in range(len(words)-win):
        ngrams.append("".join(words[cur:cur+win]))
        tok_indices.append(tok_idx[cur])
    return ngrams, tok_indices


def make_tree(dep_data, tokens=None):
    node_seq = [TreeNode(i, x) for i, x in enumerate(dep_data)]    
    root_node = TreeNode(-1, ("ROOT", "--", "ROOT", "", -1))
    
    for node in node_seq:
        # the head is not ROOT
        if node.head_idx != node.idx:
            head = node_seq[node.head_idx]
        else:
            head = root_node        
        node.parent = head
        head.children.append(node)
    
    if tokens and len(tokens) == len(dep_data):
        for idx, node in enumerate(node_seq):
            node.ckip_pos = tokens[idx][1]
    return root_node

def find_nodes(tree_node, pred_func):
    founds = []
    if pred_func(tree_node):
        founds.append(tree_node)

    if tree_node.children:
        rets = [find_nodes(x, pred_func) for x in tree_node.children]
        founds.extend(list(chain.from_iterable(rets)))

    return founds

def collect_heads(root, nodes):
    node_indices = [x.idx for x in nodes]
    head_indices = {}    
    for nd in nodes:
        if nd.head_idx == nd.idx:  # these are top nodes
            continue
        # if nd.head_idx in node_indices: # these nodes are in compounds
        #     continue
        head_indices.setdefault(nd.head_idx, []).append(nd.rel)
    

    heads = [find_nodes(root, lambda nd: nd.idx==x)[0] for x in head_indices.keys()]
    relations = ["/".join(x) for x in head_indices.values()]
    
    return heads, relations

def collect_children(node, masks=[]):
    children = [node]
    if node.children:
        rets = [collect_children(nd, masks)
                for nd in node.children
                if nd not in masks]
        children.extend(list(chain.from_iterable(rets)))
    children.sort(key=lambda x: x.idx)
    return children

def find_eval_text(tree, site_ids, mask_ids=[]):    
    nodes = find_nodes(tree, lambda x: x.idx in site_ids)
    mask_nodes = find_nodes(tree, lambda x: x.idx in mask_ids)
    site_words = ''.join([x.word for x in sorted(nodes, key=lambda x: x.idx)])
    heads, rels = collect_heads(tree, nodes)
    eval_entries = []
    for head_x, rel_x in zip(heads, rels):        
        if not re.match("VA|VB|VC|VD|VF|VH|VI|VJ|VL", head_x.ckip_pos):
            continue
        desc = collect_children(head_x, masks=mask_nodes)
        # print("{} <{}-{}>: {}".format(
        #     site_words, head_x.word, rel_x, " ".join(x.word for x in desc)))
        entry = {
            "site_words": site_words,
            "head": head_x.word,
            "evaltext": [(x.word, x.ckip_pos) for x in desc]
        }
        eval_entries.append(entry)
    
    return eval_entries
    
class TreeNode:
    def __init__(self, token_idx, token):
        self.idx = token_idx
        self.word = token[0]
        self.pos = token[1]
        self.ckip_pos = ""
        self.rel = token[2]
        self.head_idx = int(token[4])        
        self.parent = None
        self.children = []

    def __repr__(self):
        if self.word=="ROOT":
            return f"<TreeNode: Root ({len(self.children)} children)>"
        else:
            return f"<TreeNode({self.idx}): {self.word}({self.pos})>"


    def print_tree(self, depth=0):
        print("{indent}{depth}--({rel}) {word}:{idx}({pos})".format(
                indent="   "*depth, depth=depth,
                rel=self.rel, word=self.word, 
                pos=self.ckip_pos if self.ckip_pos else self.pos, 
                idx=self.idx
        ))

        for ch in self.children:
            ch.print_tree(depth=depth+1)

    def depth(self, root):
        counter = 1
        if self.idx == self.head_idx:
            return counter
        buf = find_nodes(root, lambda x: x.idx==self.head_idx)[0]        

        while buf:
            buf = find_nodes(root, lambda x: x.idx==buf.head_idx)[0]            
            counter += 1
            if buf.idx == buf.head_idx:
                return counter
            if counter > 10:
                return counter