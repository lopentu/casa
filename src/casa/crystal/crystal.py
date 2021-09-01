import re
from pathlib import Path
from collections import Counter
import pandas as pd
from copy import deepcopy

ATTR_PRIORS = {
    "[資費]方案活動": 0.05,
    "[資費]低資費方案": -0.05,
    "[加值]國際漫遊": -0.1,
    "[加值]電信APP": -0.1,    
}

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
        onto_data.sort_values(by="candidate", key=lambda x: -x.str.len(), inplace=True)
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

        detected = set()
        for eval_key, eval_val in self.eval_onto.items():
            if eval_key in text and eval_val[0][2] > 0.1:
                # if there is already a longer match,
                # skip the current match
                if any((eval_key in det_x)
                        for det_x in detected):
                    continue
                detected.add(eval_key)
                result["onto"].append((eval_key, eval_val))

        return result

    def dynamic_solve_ontos(self, ontos):
        # ontos: List[Onto]
        # Onto: Tuple[word, List[EvalParam]]
        # EvalParam: Attribute, Pol, Weight

        # trace: List[word, State]
        # State: Dict[Attribute, Score]
        trace = []
        for word, eval_params in ontos:
            _, state = trace[-1] if trace else [None, {}]
            state = deepcopy(state)
            # add category smoothing
            categories = [x[1:x.index("]")]
                          for x, _, _ in eval_params]
            for attr, attr_score in state.items():
                attr_cat = attr[1:attr.index("]")]
                if attr_cat in categories:
                    state[attr] = state.get(attr, 0.) + 0.05

            # add weighting in current eval_params
            for attr, _, w in eval_params:
                score = w + state.get(attr, 0)
                score += ATTR_PRIORS.get(attr, 0)
                state[attr] = score

            trace.append([word, state])

        # decode trace
        word_attr_map = {}
        for trace_x, onto_x in zip(trace, ontos):
            word, state = trace_x
            _, eval_params = onto_x

            max_score = max(state.values())
            max_attr = [attr for attr, score
                        in state.items()
                        if score==max_score]
            max_attr = max_attr[0] if max_attr else ""
            sel_eval_param = [x for x in eval_params if x[0]==max_attr]
            if sel_eval_param:
                word_attr_map[word] = sel_eval_param[0]

        return word_attr_map, trace

    def analyze(self, text):
        dres = self.detect(text)
        ontos = dres.get("onto", [])
        
        # attr_map: Dict[attr, (weighted_polarity_sum, weights)]
        attr_map = {}        
        word_attr_map, _ = self.dynamic_solve_ontos(ontos)
        for word, eval_param in word_attr_map.items():
            attr, pol, w = eval_param
            attr_score = attr_map.get(attr, (0.,0.))
            pol_wsum = attr_score[0] + pol*w
            weights = attr_score[1] + w
            attr_map[attr] = (pol_wsum, weights)

        if attr_map:
            top_attr = sorted(attr_map.keys(), key=lambda x: -attr_map[x][1])
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
            "result": result, "word_attr_map": word_attr_map, **dres }

