import re
from pathlib import Path
from collections import Counter
import pandas as pd

class Crystal:
    def __init__(self):
        self.eval_onto = {}
        self.cxlist = {}

    @classmethod
    def load(cls, data_dir):
        data_dir = Path(data_dir)

        # load sentiment constrcticon
        with open(data_dir / "constructions.csv", encoding="UTF-8") as fin:            
            cxlist = []
            for ln in fin.readlines():
                toks = ln.split(",")
                if len(toks) != 2: continue
                try:
                    cxlist.append((toks[0], int(toks[1])))
                except ValueError: 
                    continue

        # load eval-ontology
        onto_data = pd.read_csv(data_dir / "eval_ontology.csv", index_col=None)
        eval_onto = {}
        for row in onto_data.itertuples():
            word = row.candidate
            if len(word) == 1: continue            
            eval_onto.setdefault(word, []).append(
                (row.attribute, row.polarity,
                 int(row.relatedness)))
        for key, value in eval_onto.items():            
            eval_item = sorted(value, key=lambda x: -x[2])
            rel_sum = sum(x[2] for x in eval_item)
            eval_item = [(*x[:-1], x[2]/rel_sum) for x in eval_item]
            eval_onto[key] = eval_item
            
        inst = Crystal()
        inst.eval_onto = eval_onto
        inst.cxlist = cxlist
        return inst
    
    def detect(self, text):
        result = {"CxG": [], "onto": []}
        for cx in self.cxlist:
            mat = re.search(cx[0], text)
            if mat:
                result["CxG"].append(cx)
            
        for eval_key, eval_val in self.eval_onto.items():
            if eval_key in text and eval_val[0][2] > 0.51:                
                result["onto"].append((eval_key, *eval_val[0]))

        return result
    
    def analyze(self, text):
        dres = self.detect(text)
        ontos = dres.get("onto", [])
        attr_map = {}
        # attr_map: Dict[attr, (weighted_polarity_sum, weights)]
        for onto_x in ontos:            
            val = attr_map.get(onto_x[1], (0, 0))
            val = (val[0]+onto_x[2]*onto_x[3], val[1]+onto_x[3])
            attr_map[onto_x[1]] = val
        
        if attr_map:
            top_attr = sorted(attr_map.keys(), key=lambda x: x[2])
            top_attr_item = attr_map[top_attr[0]]
            weighted_pol = top_attr_item[0] / top_attr_item[1]
            result = (top_attr[0], weighted_pol)
        else:
            cxg_res = dres.get("CxG", [])
            if cxg_res:
                result = (None, cxg_res[0][1])
            else:
                result = (None, None)
        
        return {
            "result": result, "eval_onto": attr_map, **dres }

        