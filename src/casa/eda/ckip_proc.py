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
        sp = self.sp        
        opp.text_tokens = self.preproc(opp.text)
        opp.title_tokens = self.preproc(opp.title)
        opp.proc_info = {"segmentation": {"type": "ckip"}, "pos": {"type": "ckip"}}
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
        return tokens_list

