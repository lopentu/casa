import re
from pathlib import Path
from gensim.models.keyedvectors import KeyedVectors
import sentencepiece as spm
from .seed_lexicon import SeedLexicon
import numpy as np

def transform_scores(x):

    if np.any(x>0.99):
        # there is/are exact hits in the scores, use winner-takes-all strategy
        x_trans = x.copy()
        x_trans[x>0.99] = 5*x_trans[x>0.99]
    else:
        # transform raw scores with bias set to 0.6 and scaling factor to 5
        x_trans = 5*(x-0.6)

    # softmax
    probs = np.exp(x_trans) / np.exp(x_trans).sum()
    return probs

class Cadet:
    def __init__(self, sp, kv, lexicon, max_len=1000):
        self.kv = kv
        self.sp = sp
        self.lexicon = lexicon
        self.seed_idxs = []
        self.max_len = max_len
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

    def reduce_scores(self, mat):
        keyword_mask = (mat > 0.99).any(axis=0)
        vec = np.zeros(mat.shape[1], dtype=np.float)
        vec[keyword_mask] = (mat[:, keyword_mask]>0.99).sum(axis=0)
        vecsim_mask = np.logical_not(keyword_mask)
        vec[vecsim_mask] = mat[:, vecsim_mask].max(axis=0)
        
        return vec


    def detect(self, text, level=-1, topn=5, verbose=False):
        text = text[:self.max_len]
        sim_mat = self.build_seedsims_matrix(text, verbose)
        simvec = self.reduce_scores(sim_mat)
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
        seed_scores = np.array(simvec[srv_seeds_idxs])
        srv_seeds_probs = transform_scores(seed_scores)
        srv_seeds_order = np.argsort((-srv_seeds_probs))

        if verbose:
            print("ent_scores", ent_scores)
            print("srv_scores", srv_scores)
            print("seed_scores(topn)", seed_scores[srv_seeds_order][:topn])

        return {
            "entity": ent_labels,
            "entity_probs": ent_probs,
            "service": srv_labels,
            "service_probs": srv_probs,
            "seeds": [srv_seeds[i] for i in srv_seeds_order[:topn]],
            "seed_probs": srv_seeds_probs[srv_seeds_order][:topn]
        }

    def tokenize(self, text, verbose=False):
        stopped = "▁、。?,與和的是用跟到"
        pat1 = re.compile(f"^[{stopped}]+")
        pat2 = re.compile(f"[{stopped}]+$")

        def preproc(x):
            if len(x) <= 1:
                return x
            else:
                return pat1.sub("", pat2.sub("", x))

        tokens = self.sp.encode(text.lower().strip(), out_type=str)
        tokens = [preproc(x) for x in tokens]
        if verbose:
            print("tokens: ", tokens)
        return tokens

    def build_seedsims_matrix(self, text, verbose=False):
        tokens = self.tokenize(text, verbose)
        tokens_matrix = np.vstack(
                        [self.kv.get_vector(x)
                         for x in tokens
                         if x])
        seed_sims = tokens_matrix.dot(self.seeds_matrix.T)        

        # collapse full candidates matrix to seed matrix
        n_tokens = tokens_matrix.shape[0]
        n_seeds = len(self.lexicon.seeds)
        seed_idxs = self.seed_idxs
        Z = np.zeros((n_tokens, n_seeds), dtype=np.float)
        for i in range(np.max(seed_idxs)+1):
            mask = (seed_idxs == i)
            if np.any(mask):                                
                Z[:, i] = np.max(seed_sims[:, mask], axis=1)

        return Z