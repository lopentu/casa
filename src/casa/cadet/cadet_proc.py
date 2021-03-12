from ..opinion_types import Opinion
from ..proc_opinions import OpinionProc

class CadetProcessor():
    def __init__(self, cadet):
        self.cadet = cadet        
        self.flag = "cadet"

    def process(self, opinion: Opinion):
        opp = OpinionProc.from_opinion(opinion)
        cadet = self.cadet
        try:
            ret = cadet.detect(opp.text)
            if opp.post_type == "主文" and opp.title:
                title_ret = cadet.detect(opp.title)
                setattr(opp, "cadet_title", title_ret)
        except Exception:
            ret = {}
        setattr(opp, "cadet_result", ret)
        return opp