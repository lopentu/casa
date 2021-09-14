import numpy as np


def find_all_pos(text, target, start=0):
    try:
        pos = text.index(target, start)
        return [pos] + find_all_pos(text, target, start=pos+1)
    except ValueError:
        return []

def pn_mask(token_logits, othres=3):
    pn_mask = token_logits[:, 0] > othres
    pn_logits = token_logits[:, 1:3]
    pn_idx = pn_logits.argmax(axis=1)
    pn_prob = np.exp(pn_logits)/np.exp(pn_logits).sum(axis=1)[:, np.newaxis]
    pn_idx[pn_mask] = -1
    return pn_prob, pn_idx

def get_cadet_token_index(cadence_output):
    out = cadence_output
    text = out.text
    tok_attribs = out.cadet["tokens_attrib"]
    cadet_toks = out.cadet["tokens"]
    entity_names = out.cadet["entity"]
    cadet_idx = np.zeros(len(text), dtype=np.int32)
    attribs = list(tok_attribs.items())

    for attrib, tok_idxs in attribs[::-1]:
        for tok_idx in tok_idxs:
            tok = cadet_toks[tok_idx]
            idxs = find_all_pos(text, tok)

            if attrib in entity_names:
                code = 90
            else:
                code = 91

            for idx in idxs:
                cadet_idx[idx:idx+len(tok)] = code
    return cadet_idx

def get_crystal_token_index(cadence_output):
    out = cadence_output
    text = out.text
    word_attr_map = out.crystal.get("word_attr_map", {})    
    crystal_idx = np.zeros(len(text), dtype=np.int32) - 1
    for word, eval_params in word_attr_map.items():
        polarity = eval_params[1]

        # skip neutral word
        if polarity == 3: continue
        idxs = find_all_pos(text, word)

        for idx in idxs:
            crystal_idx[idx:idx+len(word)] = int(polarity<3)

    return crystal_idx

def visualize_tokens(cadence_output, othres=4, quiet=False):
    token_logits = cadence_output.mt_bert["opn_logits"][1:-1]
    text = cadence_output.text
    pn_prob, pn_idx = pn_mask(token_logits, othres)
    cadet_idx = get_cadet_token_index(cadence_output)
    crystal_idx = get_crystal_token_index(cadence_output)

    vistext = ""

    def _print(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)        

    for tok_i, tok in enumerate(text.replace(" ", "")):
        if tok_i >= pn_idx.size: break

        if cadet_idx[tok_i] == 90:
            # blue
            vistext += f"\x1b[34m{tok}\x1b[0m"
        elif cadet_idx[tok_i] == 91:
            # green
            vistext += f"\x1b[32m{tok}\x1b[0m"                
        elif crystal_idx[tok_i] == 1:
            # red
            vistext += f"\x1b[31m{tok}\x1b[0m"
        elif crystal_idx[tok_i] == 0:
            # cyan
            vistext += f"\x1b[36m{tok}\x1b[0m"
        elif pn_idx[tok_i] == 1:
            # red
            vistext += f"\x1b[31m{tok}\x1b[0m"
        elif pn_idx[tok_i] == 0:
            # cyan
            vistext += f"\x1b[36m{tok}\x1b[0m"
        else:
            vistext += tok
        if (tok_i+1) % 60 ==0:
            _print(vistext)
            vistext = ""
    _print(vistext)

    pn_idx[cadet_idx!=0] = cadet_idx[cadet_idx!=0]
    vis_tokens = {"text": text, 
                  "tag_idx": pn_idx,
                  "pn_prob": pn_prob}
    return vis_tokens