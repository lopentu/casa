def pn_mask(token_probs, pn_thres=0.1):
    pn_prob, pn_idx = token_probs[:, 1:3].max(dim=-1)
    pn_idx[pn_prob < pn_thres] = -1
    return pn_prob.numpy(), pn_idx.numpy()

def visualize_tokens(cadence_output, pn_thres=0.1):
    token_probs = cadence_output.mt_bert["token_probs"]
    text = cadence_output.mt_bert["text"]
    pn_prob, pn_idx = pn_mask(token_probs, pn_thres)

    vistext = ""

    for tok_i, tok in enumerate(text.replace(" ", "")):
        if tok_i >= pn_idx.size: break
        if pn_idx[tok_i] == 1:
            # red
            vistext += f"\x1b[31m{tok}\x1b[0m"
        elif pn_idx[tok_i] == 0:
            # green
            vistext += f"\x1b[32m{tok}\x1b[0m"
        else:
            vistext += tok
        if (tok_i+1) % 60 ==0:
            print(vistext)
            vistext = ""
    print(vistext)