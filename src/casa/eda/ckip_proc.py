import logging
from itertools import islice
from tqdm.auto import tqdm
from .base_proc import EdaProcessor
from ..opinion_types import Opinion
from ..proc_opinions import OpinionProc

class CkipSubmitProcessor(EdaProcessor):
    def __init__(self, ckip_proxy):
        self.ckip_proxy = ckip_proxy 
        self.flag = ""    

    def process(self, opinion: Opinion):        
        self.ckip_proxy.submit(opinion.id+"h", opinion.title)
        self.ckip_proxy.submit(opinion.id+"x", opinion.text)

        return opinion        
class CkipRetrieveProcessor(EdaProcessor):
    def __init__(self, ckip_proxy):
        self.ckip_proxy = ckip_proxy
        self.flag = "ckip"

    def process(self, opinion: Opinion):
        opp = OpinionProc.from_opinion(opinion)        
        text_tokens = self.ckip_proxy.query(opp.id+"x")
        title_tokens = self.ckip_proxy.query(opp.id+"h")

        if text_tokens:
            opp.text_tokens = [x[0] for x in text_tokens]
        if title_tokens:
            opp.title_tokens = [x[0] for x in title_tokens]

        if text_tokens or title_tokens:
            opp.proc_info.update({
                "segmentation": {"type": "ckip"}, 
                "pos": {"type": "ckip"},
                "ckip": {"text": text_tokens, "title": title_tokens}})
            return opp
        else:
            return opinion

class CkipProxy:
    def __init__(self, ws, pos, batch_size=32):
        self.ws = ws
        self.pos = pos
        self.buf = {}
        self.done = {}        
        self.batch_size = batch_size

    def submit(self, task_id, text):
        self.buf[task_id] = text
    
    def query(self, task_id):
        return self.done.get(task_id)
    
    def batch(self, iterable, n=32):
        iterator = iter(iterable)
        while True:
            batch = list(islice(iterator, n))
            if batch:
                yield batch
            else:
                break

    def process(self):        
        n_batch = (len(self.buf) // self.batch_size) + 1
        for batch in tqdm(self.batch(self.buf.items()), total=n_batch):
            task_ids, texts = zip(*batch)
            tokens_list = self.tokenize(texts)

            if len(tokens_list) != len(task_ids):
                logging.error("tokenization length mismatch")
                continue

            for idx, tokens in enumerate(tokens_list):
                task_id = task_ids[idx]                
                self.done[task_id] = tokens


    def tokenize(self, text):
        if isinstance(text, str):
            text = [text]
        ws_list = self.ws(text)
        pos_list = self.pos(ws_list)

        tokens_list = []
        for i in range(len(text)):            
            tokens = []
            for word, pos in zip(ws_list[i], pos_list[i]):
                tokens.append((word, pos))
            tokens_list.append(tokens)
        
        # squeeze output
        if len(tokens_list) == 1 and isinstance(tokens_list[0], list):
            tokens_list = tokens_list[0]
        return tokens_list
