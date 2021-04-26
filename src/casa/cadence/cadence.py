import re
import numpy as np
import torch

class Cadence:
    def __init__(self):
        pass
    
    @classmethod
    def build_Q2(cls, cadet, pred_model, tokenizer, cx_list):
        inst = Cadence()
        setattr(inst, "cadet", cadet)
        setattr(inst, "model", pred_model)
        setattr(inst, "tokenizer", tokenizer)
        setattr(inst, "cx_list", cx_list)
        return inst
        
    def sentiment_bert(self, intxt):
        X = self.tokenizer(intxt, return_tensors="pt")
        probs = None
        with torch.no_grad():
            out = self.model(**X)
            probs = torch.softmax(out.logits, axis=1).squeeze().numpy()        
        return probs

    def sentiment_CxG(self, intxt):
        out_probs = None
        for cx, score in self.cx_list:        
            pat = cx.replace("\b", "\\b")
            if re.match(pat, intxt):
                is_positive = int(score>3)
                out_probs = np.zeros(3, dtype=np.float32)
                out_probs[2-is_positive] = 1.
                break    
        return out_probs
    
    def sentiment(self, intxt, mode="all"):
        src = "CxG"
        labels = ["Neutral", "Positive", "Negative"]
        probs = None
        
        if mode.lower() != "bert_only":
            probs = self.sentiment_CxG(intxt)        
        
        if probs is None and mode.lower()!="cxg_only":
            probs = self.sentiment_bert(intxt)
            src = "Bert"
            
        if probs is None:
            probs = np.array([1, 0, 0], dtype=np.float32)
            src = "Abstain"
        
        return {"sentiment": labels,
                "sentiment_src": src, 
                "sentiment_probs": probs}
    
    def analyze(self, intxt, sentiment_mode="all", summary=False):
        dets = self.cadet.detect(intxt)
        sentiments = self.sentiment(intxt, sentiment_mode)
        out = {**dets, **sentiments}
        if summary:
            return self.summary(out)
        else:
            return out
    
    def summary(self, x):
        (E, A, P) = [""] * 3
        (Ep, Ap, Pp) = [0.] * 3
        Psrc = "NA"
        if "entity" in x:
            E = x["entity"][np.argmax(x["entity_probs"])]
            Ep = np.max(x["entity_probs"])
        if "service" in x:
            A = x["service"][np.argmax(x["service_probs"])]
            Ap = np.max(x["service_probs"])
        if "sentiment" in x:
            P = x["sentiment"][np.argmax(x["sentiment_probs"])]
            Psrc = x["sentiment_src"]
            Pp = np.max(x["sentiment_probs"])
        return f"{E}({Ep:.2f})/{A}({Ap:.2f})/{P}({Pp:.2f}, {Psrc})"