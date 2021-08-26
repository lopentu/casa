import json
import re
import numpy as np
import torch
from ..cadet import Cadet
from ..crystal import Crystal
from ..MTBert import MTBert
from .resolvers import (
    CadenceOutput,
    CadenceResolveStrategy,
    CadenceSimpleResolver,
    CadenceMultiResolver,
    CadenceBertOnlyResolver)

class Cadence:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="UTF-8") as fin:
            config = json.load(fin)
        self.cadet = Cadet.load(config.cadet_path)
        self.crystal = Crystal.load(config.crystal_path)
        self.mt_bert = MTBert.load(config.mtbert_path)
    
    @classmethod
    def load(cls, config_path):
        inst = Cadence(config_path)
        return inst
            
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
    
    def process(self, intxt: str) -> CadenceOutput:
        cadet_res = self.cadet.detect(intxt)
        crystal_res = self.crystal.analyze(intxt)
        mtbert_res = self.mt_bert.analyze(intxt)
        
        out = CadenceOutput(cadet_res, crystal_res, mtbert_res)
        return out

    def analyze(self, intxt, 
                strategy=CadenceResolveStrategy.Simple):
        out = self.process()
        if strategy == CadenceResolveStrategy.Simple:
            return CadenceSimpleResolver().resolve(out)
        if strategy == CadenceResolveStrategy.Multiple:
            return CadenceMultiResolver().resolve(out)
        if strategy == CadenceResolveStrategy.BertOnly:
            return CadenceBertOnlyResolver().resolve(out)
        else:
            raise ValueError("Unsupported strategy")
        