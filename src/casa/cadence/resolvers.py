from enum import Enum, auto
from re import L
import numpy as np
from . import multi_resolver_utils as mr

class CadenceResolveStrategy(Enum):
    Simple   = auto()
    Multiple = auto()
    BertOnly = auto()    

class CadenceOutput:
    def __init__(self, cadet_res, crystal_res, mt_bert_res, **kwargs):
        self.cadet = cadet_res
        self.crystal = crystal_res
        self.mt_bert = mt_bert_res
        self.aspects = []
        self.flags = kwargs
    
    def format_aspect(self):
        asp_fmt = []
        for asp_x in self.aspects:
            ent_txt = asp_x[0] if asp_x[0] else "無特定業者"
            srv_txt = asp_x[1] if asp_x[1] else "無特定服務"
            asp_fmt.append((ent_txt, srv_txt, asp_x[2]))
        return asp_fmt

    def __repr__(self):
        return f"<CadenceOutput: {self.format_aspect()}>"
    
    @property
    def entities(self):
        ent_iter = filter(lambda x: x in self.cadet.get("entity", []),
                    self.cadet.get("tokens_attrib", {}).keys())
        return list(ent_iter)

    @property
    def text(self):
        return self.mt_bert["text"].replace(" ", "")

class CadetResolverMixin:
    def resolve_cadet(self, cadet_output, 
                      entity_thres=0.3, service_thres=0.1):
        out = cadet_output
        ent_probs = out["entity_probs"]
        srv_probs = out["service_probs"]
        ent_list = out["entity"]
        srv_list = out["service"]

        top_ent = np.argmax(ent_probs)
        top_srv = np.argmax(srv_probs)
        ent_maxp = np.max(ent_probs)
        srv_maxp = np.max(srv_probs)        
        
        ret_ent = ent_list[top_ent] if ent_maxp > entity_thres else None
        ret_srv = srv_list[top_srv] if srv_maxp > service_thres else None
        return (ret_ent, ret_srv)

class CadenceSimpleResolver(CadetResolverMixin):
    def __init__(self, crystal_text_len=500, mtbert_neu_prob_thres=0.7):
        # maximum length to pass through Crystal.
        # i.e. text longer than the threshold skip Crystal stage.
        self.crystal_text_len = crystal_text_len
        self.mtbert_neu_prob_thres = mtbert_neu_prob_thres
    
    def resolve(self, out: CadenceOutput):                
        op_crystal = out.crystal["result"]
        op_cadet = out.cadet
        op_mtbert = out.mt_bert
        cadet_det = self.resolve_cadet(op_cadet)
        # aspect: List[entity, service, polarity, aspect_source, polarity_source]
        fASP_SRC = 3
        fPOL_SRC = 4
        aspect = [cadet_det[0], cadet_det[1], -1, "cadet", "none"]

        # use crystal result if available
        if op_crystal[0]:
            aspect[1] = op_crystal[0]
            aspect[fASP_SRC] = "crystal"
        if op_crystal[1] is None:
            # op_crystal[1] is None, crystal has no say
            pass
        elif op_crystal[1] > 3:
            aspect[2] = "Positive"
            aspect[fPOL_SRC] = "crystal"
        elif op_crystal[1] < 3:
            aspect[2] = "Negative"
            aspect[fPOL_SRC] = "crystal"
        else:
            # leave the aspect polarity untouched
            pass

        # use mt_bert seq result
        BYPASS_CRYSTAL = len(out.text) > self.crystal_text_len
        if (BYPASS_CRYSTAL or aspect[2] == -1 or 
            op_mtbert["seq_probs"][0] > self.mtbert_neu_prob_thres):
            aspect[2] = op_mtbert["seq_polarity"]
            aspect[fPOL_SRC] = "mtbert"
        
        return [aspect]

class CadenceMultiResolver(CadetResolverMixin):
    def __init__(self):
        self.label_map = None
        self.ch_labels = None
        self.aspects = None
        self.cadence_out = None
    
    def resolve(self, out: CadenceOutput):  
        label_map = mr.compute_label_maps(out)
        ch_labels = mr.build_char_labels(label_map)
        aspects = mr.build_aspects(out, ch_labels)
        self.cadence_out = out
        self.label_map = label_map
        self.ch_labels = ch_labels
        self.aspects = aspects
        return aspects


class CadenceBertOnlyResolver(CadetResolverMixin):
    def __init__(self):
        pass
    
    def resolve(self, out: CadenceOutput):                    
        op_cadet = out.cadet
        op_mtbert = out.mt_bert
        cadet_det = self.resolve_cadet(op_cadet)
        # aspect: List[entity, service, polarity, aspect_source, polarity_source]
        fASP_SRC = 3
        fPOL_SRC = 4
        aspect = [cadet_det[0], cadet_det[1], -1, "cadet", "none"]

        # skip crystal result
        
        # use mt_bert seq result
        aspect[2] = op_mtbert["seq_polarity"]
        aspect[fPOL_SRC] = "mtbert"                

        return [aspect]