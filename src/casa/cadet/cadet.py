import re
from pathlib import Path
from gensim.models.keyedvectors import KeyedVectors
import sentencepiece as spm
from .seed_lexicon import SeedLexicon
import numpy as np

def transform_scores(x):
    # logistic function with bias set to 0.6 and scaling factor 10
    if np.any(x>0.99):
        # there is/are exact hits in the scores, use winners takes all strategy
        x_trans = x.copy()
        x_trans[x>0.99] = 10
    else:
        x_trans = 1/(1+np.exp(-10*(x-0.6)))
    
    # softmax
    probs = np.exp(x_trans) / np.exp(x_trans).sum()
    return probs

class Cadet:
    def __init__(self, sp, kv, lexicon):
        self.kv = kv
        self.sp = sp
        self.lexicon = lexicon
        self.seed_idxs = []
        self.build_seed_matrix()

    def __repr__(self):
        ft_dim = str(self.kv.vectors_vocab.shape)
        n_seed = len(self.lexicon.seeds)        
        return f"<Cadet: FastText{ft_dim}, Seeds({n_seed})>"

    @classmethod
    def load(self, base_dir):
        base_dir = Path(base_dir)
        lexicon = SeedLexicon.load(base_dir / "seeds.csv")        
        sp = spm.SentencePieceProcessor(model_file=str(base_dir/"spm-2020.model"))
        kv = KeyedVectors.load(str(base_dir/"ft-2020.kv"))
        kv.init_sims(replace=True)
        return Cadet(sp, kv, lexicon)

    def build_seed_matrix(self):
        candidates, seed_idxs = self.lexicon.flatten_candidates()
        self.seeds_matrix = np.vstack(
            [self.kv.get_vector(x) for x in candidates])
        self.seed_idxs = np.array(seed_idxs)
    
    def detect(self, text, level=-1):        
        simvec = self.build_seedsims_matrix(text).max(axis=0)
        ent_idxs = self.lexicon.get_entities()
        ent_labels = list(ent_idxs.keys())
        ent_scores = np.array([simvec[idxs].max() 
                            for idxs in ent_idxs.values()])
        ent_probs = transform_scores(ent_scores)

        srv_idxs = self.lexicon.get_services(level)
        srv_labels = list(srv_idxs.keys())
        srv_scores = np.array([simvec[idxs].max() 
                        for idxs in srv_idxs.values()])            
        srv_probs = transform_scores(srv_scores)

        srv_seeds, srv_seeds_idxs = self.lexicon.get_service_seeds()
        srv_seeds_idxs = np.array(srv_seeds_idxs)
        seed_scores = simvec[srv_seeds_idxs]                
        srv_seeds_probs = transform_scores(seed_scores)        
        srv_seeds_order = np.argsort((-srv_seeds_probs))        

        return {
            "entity": ent_labels,
            "entity_probs": ent_probs,
            "service": srv_labels,
            "service_probs": srv_probs,
            "seeds": [srv_seeds[i] for i in srv_seeds_order[:5]],
            "seed_probs": srv_seeds_probs[srv_seeds_order][:5] 
        }

    def tokenize(self, text):
        pat1 = re.compile("^[、。與的是]+")
        pat2 = re.compile("[、。與的是]+$")
                
        tokens = self.sp.encode(text.lower().strip(), out_type=str)
        tokens = [pat1.sub("", pat2.sub("", x)) for x in tokens]
        return tokens

    def build_seedsims_matrix(self, text):
        tokens = self.tokenize(text)
        print("tokens: ", tokens)
        tokens_matrix = np.vstack(
                        [self.kv.get_vector(x) 
                         for x in tokens])
        seed_sims = tokens_matrix.dot(self.seeds_matrix.T)

        # collapse matrix
        n_tokens = len(tokens)
        n_seeds = len(self.lexicon.seeds)
        seed_idxs = self.seed_idxs
        Z = np.zeros((n_tokens, n_seeds), dtype=np.float)
        for i in range(np.max(seed_idxs)+1):
            mask = (seed_idxs == i)            
            if np.any(mask):                
                Z[:, i] = np.max(seed_sims[:, mask], axis=1)
        
        return Z