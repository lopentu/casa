import json
import re
import numpy as np
from pathlib import Path
import torch
import logging
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
        logger = logging.getLogger("casa.Cadence")
        with open(config_path, "r", encoding="UTF-8") as fin:
            config = json.load(fin)     
        base_dir = config_path.parent   
        logger.info("Loading Cadet")
        self.cadet = Cadet.load(base_dir/config["cadet_path"])        

        logger.info("Loading Crystal")
        self.crystal = Crystal.load(base_dir/config["crystal_path"])

        logger.info("Loading MTBert")
        self.mt_bert = MTBert.load(base_dir/config["mtbert_path"])
    
    @classmethod
    def load(cls, config_path):
        config_path = Path(config_path)
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
    
    def process(self, intxt: str, max_eval_count=3) -> CadenceOutput:
        cadet_res = self.cadet.detect(intxt)
        crystal_res = self.crystal.analyze(intxt, max_eval_count=max_eval_count)
        mtbert_res = self.mt_bert.analyze(intxt)
        
        out = CadenceOutput(cadet_res, crystal_res, mtbert_res)
        return out

    def analyze(self, intxt, strategy="simple", max_eval_count=3):
        out = self.process(intxt, max_eval_count=max_eval_count)
        if strategy.lower() == "simple":
            out.aspects = CadenceSimpleResolver().resolve(out)
        elif strategy.lower() == "multiple":
            out.aspects = CadenceMultiResolver().resolve(out)
        elif strategy.lower() == "bertonly":
            out.aspects = CadenceBertOnlyResolver().resolve(out)
        else:
            raise ValueError("Unsupported strategy: must be one of simple, multiple, bertonly")
        return out