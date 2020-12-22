from itertools import chain

class TreeNode:
    def __init__(self, token_idx, token):        
        self.idx = token_idx
        self.word = token[0]
        self.pos = token[1]
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
                rel=self.rel, word=self.word, pos=self.pos, idx=self.idx
        ))
        
        for ch in self.children:
            ch.print_tree(depth=depth+1)

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


def make_tree(dep_data):
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
    return root_node