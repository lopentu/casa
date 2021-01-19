from gensim.models import KeyedVectors
from typing import List
import numpy as np

class SeedEntry:
    def __init__(self, category, seed, candidates):
        self.category = category
        self.seed = seed
        self.candidates = candidates

    def __repr__(self):
        return (f"<SeedEntry: {self.category}/{self.seed}"
                f"({len(self.candidates)} candidates)>")

class SeedDistribution:
    def __init__(self, spvec:KeyedVectors,
            entity_seeds: List[SeedEntry],
            service_seeds: List[SeedEntry]):
        self.spvec = spvec
        self.entity_seeds = entity_seeds
        self.service_seeds = service_seeds
        self.seed_matrix = None
        self.entity_cursors = []
        self.service_cursors = []
        self.__build_seed_matrix(spvec, entity_seeds, service_seeds)

    def get_distribution(self, text, window=3):
        # text_matrix: n_ngram x dim
        spvec = self.spvec
        n_ngram = len(text)-window+1
        dim = spvec.vectors.shape[1]
        text_matrix = np.zeros((n_ngram, dim), dtype=np.double)

        for cur in range(0, len(text)-window+1):
            ngram = text[cur:cur+window]
            ng_vec = spvec.get_vector(ngram)
            text_matrix[cur, :] = ng_vec

        candid_matrix = self.seed_matrix.dot(text_matrix.T)
        topic_matrix = self.__collapse_matrix(candid_matrix)
        return topic_matrix


    def __build_seed_matrix(self, spvec, 
                entity_seeds, service_seeds):
        """
        build seed_matrix and seed_cursors. This function directly set the
        object properties.

        Fields
        -------

        seed_matrix: (n_topic, dim)
                     the vector representation of each candidates in seed entries

        entity_cursors: List[Tuple[int, int]], len(entity_seeds)
                      the cursors indicating the range of vectors representing each
                      candidates in seed_matrix
        
        service_cursors: List[Tuple[int, int]], len(service_seeds)
                      the cursors indicating the range of vectors representing each
                      candidates in seed_matrix

        Arguments
        ----------
        spvec: `gensim.models.KeyedVectors`

        entity_seeds: List[SeedEntry]

        service_seeds: List[SeedEntry]
        """

        n_ent_candids = sum(len(x.candidates) for x in entity_seeds)
        n_serv_candids = sum(len(x.candidates) for x in service_seeds)

        n_row = n_ent_candids + n_serv_candids
        M = np.zeros((n_row, spvec.vectors.shape[1]), dtype=np.double)
        entity_cursors = []
        service_cursors = []
        row_counter = 0
        
        for entry_x in (entity_seeds + service_seeds):
            entry_start = row_counter
            for candid_x in entry_x.candidates:
                M[row_counter, :] = spvec.get_vector(candid_x)
                row_counter += 1
            entry_end = row_counter

            if row_counter < n_ent_candids:
                entity_cursors.append((entry_start, entry_end))
            else:
                service_cursors.append((entry_start, entry_end))
        self.seed_matrix = M
        self.entity_cursors = entity_cursors
        self.service_cursors = service_cursors

    def __collapse_matrix(self, candid_matrix):        
        ent_cursors = self.entity_cursors
        serv_cursors = self.service_cursors

        


