import re
from itertools import chain
import unicodedata as ud
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import torch
from icecream import ic

def custom_clustering(inst, s, pat, tagger_inst):
    inst._update_s_cache(s)
    out = tagger_inst.soft_tag(s)
    wlogits = torch.cat(out[0])
    tokens = inst._segments_s    
    out_tokens = list(chain(*out[2]))
    wprobs = torch.softmax(wlogits, axis=1).numpy()[:,0]    
    p_wb_tokens = np.ones(len(tokens))
    
    cur = [0, 0]
    buf = ""    
    align_tokens = []
    align_s2t = [(0,0)] * len(tokens)
    align_t2s = np.ones(len(out_tokens)+1, dtype=np.int32) * (len(tokens)-1)
    for token_i, token_s in enumerate(inst._segments_s):
        if token_s == '':
            # ic("skip empty token")
            continue
        while cur[1] < len(out_tokens):
            token_t = ud.normalize("NFKC", out_tokens[cur[1]])
            buf += token_t            
            # ic(token_s, buf)
            if buf == token_s:
                p_wb_tokens[token_i] = wprobs[cur[0]:cur[1]+1].max()
                align_tokens.append(buf)
                align_s2t[token_i] = (cur[0], cur[1]+1)
                align_t2s[cur[0]:cur[1]+1] = token_i
                buf = ''            
                cur = [cur[1]+1, cur[1]+1]
                break
            else:
                cur = [cur[0], cur[1]+1]
    
    if buf:
        align_tokens.append(buf)
        align_s2t[token_i] = (cur[0], cur[1]+1)
        align_t2s[cur[0]:cur[1]+1] = token_i
    
    mat = re.search(pat, s)    
    pat_spans = []
    if mat:
        pat_spans = [mat.span(x) for x in range(1, len(mat.groups())+1)]
    
    def in_pat(s_idx):        
        t_span = align_s2t[s_idx]            
        if t_span[0] == t_span[1]:
            # reject empty span
            return False
        
        for pat_x in pat_spans:
            if t_span[0] >= pat_x[0] and t_span[1] <= pat_x[1]:
                return (align_t2s[pat_x[0]], align_t2s[pat_x[1]])
        return ()
    
    def token_dist(x, y): 
        x = int(x); y = int(y)
        x = min(x, y); y = max(x, y)
        # word boundary distance is the word probability of the following character,
        # therefore a plus one offset.        
        span_x = in_pat(x)
        span_y = in_pat(y)
        if span_x and span_y and span_x != span_y:
            ic(x, span_x, y, span_y)
            x_start = min(x+1, span_x[1])
            x_end = span_x[1]
            x_rng = np.arange(x_start, x_end)
            y_start = min(span_y[0]+1, span_y[1])
            y_end = min(y+1, span_y[1])
            y_rng = np.arange(y_start, y_end)
            rng = np.concatenate([x_rng, y_rng])
            ic(rng)
            dist = p_wb_tokens[rng].sum()            
            # dist = p_wb_tokens[x+1:y+1].sum()
        elif span_x and span_x == span_y:
            span_length = span_x[1] - span_x[0]
            dist = p_wb_tokens[x+1:y+1].sum() / span_length
        else:
            dist = p_wb_tokens[x+1:y+1].sum()
        return dist
        
    # ic(align_s2t)
    # ic(align_t2s)
    # ic(p_wb_tokens)
    # ic(pat_spans)
    inX = np.arange(len(tokens)).reshape(-1, 1)
    Z = linkage(inX, metric=token_dist, optimal_ordering=True)
    return Z