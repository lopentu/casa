import sentencepiece as spm
from .base_proc import EdaProcessor
from ..opinion_types import Opinion
from ..proc_opinions import OpinionProc

class CkipProcessor(EdaProcessor):
    def __init__(self, ckip_ws, ckip_pos):
        self.ws = ckip_ws
        self.pos = ckip_pos
        self.flag = "ckip"

    def process(self, opinion: Opinion):
        opp = OpinionProc.from_opinion(opinion)        
        text_tokens = self.preproc(opp.text)        
        title_tokens = self.preproc(opp.title)

        opp.text_tokens = [x[0] for x in text_tokens]
        opp.title_tokens = [x[0] for x in title_tokens]

        opp.proc_info.update({
          "segmentation": {"type": "ckip"}, 
          "pos": {"type": "ckip"},
          "ckip": {"text": text_tokens, "title": title_tokens}})
        return opp
    
    def preproc(self, text):
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

