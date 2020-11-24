import sentencepiece as spm
from .base_proc import EdaProcessor
from ..opinion_types import Opinion
from ..proc_opinions import OpinionProc

class SpmEdaProcessor(EdaProcessor):
    def __init__(self, spm_model_path):
        self.spm_model_path = spm_model_path
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        self.flag = "spm"

    def process(self, opinion: Opinion):
        opp = OpinionProc.from_opinion(opinion)
        sp = self.sp
        model_path = self.spm_model_path
        opp.text_tokens = sp.encode(opp.text, out_type=str)
        opp.title_tokens = sp.encode(opp.title, out_type=str)
        opp.proc_info = {"tokenization": {"type": "spm", "model_path": model_path}}
        return opp