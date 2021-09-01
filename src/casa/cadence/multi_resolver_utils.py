
from typing import List, Dict
from itertools import groupby
from .utils import visualize_tokens

# LabelMap:
# ent:  {'亞太電信': [0], '中華電信': [7]}
# srv:  {'[通訊]網速': [4], '[通訊]涵蓋': [12]}
# pol:  {'Negative': [4], 'Positive': [12]}
LabelMap = Dict[str, any]
# CharMap
# {0: [('ent', '亞太電信')],
# 7: [('ent', '中華電信')],
# 4: [('srv', '[通訊]網速'), ('pol', 'Negative')],
# 12: [('srv', '[通訊]涵蓋'), ('pol', 'Positive')]}
CharMap = Dict[str, any]


def find_all_pos(text, target, start=0):
    try:
        pos = text.index(target, start)
        return [pos] + find_all_pos(text, target, start=pos+1)
    except ValueError:
        return []

def compute_label_maps(out: "CadenceOutput") -> LabelMap:
    ent_tokens = {}
    srv_tokens = {}
    pol_tokens = {}
    cadet_res = out.cadet
    crystal_res = out.crystal
    mtbert_res = out.mt_bert

    raw_text = mtbert_res.get("text", "")
    cadet_tokens = cadet_res.get("tokens", [])

    # get entity tokens
    for attrib, tok_idxs in cadet_res.get("tokens_attrib", {}).items():
        if attrib not in cadet_res.get("entity", []):
            # not an entity attribute
            continue
        for tok_idx in tok_idxs:
            tok = cadet_tokens[tok_idx]
            pos_list = find_all_pos(raw_text, tok)
            ent_tokens.setdefault(attrib, []).extend(pos_list)

    # get service tokens
    # use crystal if available
    word_attr_map = crystal_res["word_attr_map"]
    for word, attr in word_attr_map.items():
        indices = find_all_pos(raw_text, word)
        srv_tokens.setdefault(attr[0], []).extend(indices)
        pol_score = attr[1]
        if pol_score > 3:
            pol_tokens.setdefault("Positive", []).extend(indices)
        elif pol_score < 3:
            pol_tokens.setdefault("Negative", []).extend(indices)
            
    # if crystal is abstained, use cadet service tokens
    if not srv_tokens:
        for attrib, tok_idxs in cadet_res.get("tokens_attrib", {}).items():
            if attrib in cadet_res.get("entity", []):
                # skip entity attribute
                continue
            for tok_idx in tok_idxs:
                tok = cadet_tokens[tok_idx]
                pos_list = find_all_pos(raw_text, tok)
                srv_tokens.setdefault(attrib, []).extend(pos_list)

    if not pol_tokens:
        pn = visualize_tokens(out, pn_thres=0.2, quiet=True)
        pn_idx = pn["pn_idx"]    
        grp_iter = groupby(enumerate(pn_idx), key=lambda x: x[1])
        groups = [(gk, [idx for idx, _ in gv]) for gk, gv in grp_iter]    
        for pn_code, idx_list in groups:
            if pn_code < 0: continue
            pn = "Positive" if pn_code == 0 else "Negative"
            first_idx = idx_list[0]
            if pn not in pol_tokens or pol_tokens[pn][-1] < first_idx-1:
                pol_tokens.setdefault(pn, []).append(first_idx)
    
    return {"ent": ent_tokens, "srv": srv_tokens, "pol": pol_tokens}

def build_char_labels(label_map: LabelMap):
    ch_labels = {}
    def update_ch_labels(new_dict):
        for k, v in new_dict.items():
            ch_labels.setdefault(k, []).append(v)

    def index_positions(labtype, label_map):
        return {v:(labtype, k) 
                for k, vs in label_map.items() 
                for v in vs}

    update_ch_labels(index_positions("ent", label_map["ent"]))
    update_ch_labels(index_positions("srv", label_map["srv"]))
    update_ch_labels(index_positions("pol", label_map["pol"]))
    return ch_labels

def build_aspects(cadence_out: "CadenceOutput", ch_labels: CharMap):
    buf = {}
    aspects = []
    raw_text = cadence_out.mt_bert.get("text", "")
    for ch_i, ch_x in enumerate(raw_text):
        if ch_i not in ch_labels:
            continue
        for label, data in ch_labels[ch_i]:
            buf[label] = data
        if "ent" in buf and "srv" in buf and "pol" in buf:
            aspects.append((buf["ent"], buf["srv"], buf["pol"]))
            del buf["pol"]
    if not aspects:
        aspects.append((buf.get("ent"), buf.get("srv"), buf.get("pol")))
    return aspects
