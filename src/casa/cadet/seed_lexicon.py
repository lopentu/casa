import re
from itertools import chain

class SeedLexicon:
    def __init__(self, categories, seeds, candidates):
        assert len(categories)==len(seeds)==len(candidates)
        self.__categories = categories
        self.__seeds = seeds
        self.__candidates = candidates

    @classmethod
    def load(cls, seed_path):        
        topic_seeds_path = seed_path
        fin = open(topic_seeds_path, "r", encoding="UTF-8")
        columns = fin.readline().split(",")
        columns = [x.lower() for x in columns]
        seed_colidx = columns.index("seed")
        categories = []
        seeds = []
        candidates = []
        pat = re.compile(r"[\"']")
        for ln in fin:
            tokens = [pat.sub("", x) for x in ln.strip().split(",")]
            tokens = [x for x in tokens if x]
            categories.append(tokens[:seed_colidx])
            seeds.append(tokens[seed_colidx])
            candidates.append(tokens[seed_colidx+1:])    
        fin.close()

        return SeedLexicon(categories, seeds, candidates)
    
    @property
    def seeds(self):
        return self.__seeds
    
    def get_entities(self):
        entities = {}        

        for idx, (cat, seed) in enumerate(
                zip(self.__categories, self.__seeds)):
            if cat[0] != "ENTITY": 
                continue
            group_x = seed
            entities.setdefault(group_x, []).append(idx)
        return entities
    
    def get_services(self, level=-1):
        services = {}        

        for idx, (cat, seed) in enumerate(
                zip(self.__categories, self.__seeds)):
            if cat[0] != "SERVICE": 
                continue
            if level + 1 == 0:
                group_x = tuple(cat[1:])
            else:
                group_x = tuple(cat[1:level+1])
            services.setdefault(group_x, []).append(idx)
        return services

    def get_service_seeds(self):
        seeds = []
        seed_idxs = []
        for idx, (cat, seed) in enumerate(
                zip(self.__categories, self.__seeds)):
            if cat[0] != "SERVICE": 
                continue
            seeds.append(seed)
            seed_idxs.append(idx)
        
        return seeds, seed_idxs

    @property
    def candidates(self):
        return self.__candidates

    def get_seed_idx(self, seed):
        if seed not in self.__seeds:
            raise IndexError(f"Seed {seed} is not in lexicon")
        return self.seeds.index(seed)

    def get_category(self, seed, level=-1):        
        seed_idx = self.get_seed_idx(seed)
        cats = self.__categories[seed_idx]
        if level < 0:
            return cats
        else:
            return cats[level]

    def get_candidate(self, seed):
        seed_idx = self.get_seed_idx(seed)
        candidates = self.__candidates[seed_idx]
        return candidates
    
            
    def flatten_candidates(self):
        seed_idx_list = []
        candids_list = []
        for seed_idx, candids in enumerate(self.__candidates):
            seed_idx_list.extend([seed_idx] * len(candids))
            candids_list.extend(candids)
        return candids_list, seed_idx_list
        

