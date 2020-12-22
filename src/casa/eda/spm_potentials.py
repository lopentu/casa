from itertools import groupby
import numpy as np

def compute_potentials(words, seeds, wv):    
    potentials = []
    seed_vecs = np.vstack([wv.get_vector(x) for x in seeds])
    for target in words:        
        target_vec = wv.get_vector(target)
        score = np.mean(wv.cosine_similarities(target_vec, seed_vecs))
        potentials.append(score)
    return potentials

def extract_with_potentials(words, potentials, epsilon=0.3):
    tok_scores = list(zip(words, potentials))

    # find target words
    target_words = []
    target_indices = []    
    buf = []
    for tok_idx, (word, score) in enumerate(tok_scores):    
        if score > epsilon:
            buf.append((tok_idx, word))            
        else:
            if buf:
                target_words.append("".join([x[1] for x in buf]))
                target_indices.append([x[0] for x in buf])
                buf = []            
    if buf:
        target_words.append("".join([x[1] for x in buf]))
        target_indices.append([x[0] for x in buf])
    return target_words, target_indices