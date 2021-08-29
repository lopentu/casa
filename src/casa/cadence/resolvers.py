from enum import Enum, auto
import numpy as np

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
    def __init__(self):
        pass
    
    def resolve(self, out: CadenceOutput):                
        op_crystal = out.crystal["result"]
        op_cadet = out.cadet
        op_mtbert = out.mt_bert
        cadet_det = self.resolve_cadet(op_cadet)
        aspect = [cadet_det[0], cadet_det[0], -1]

        # use crystal result if available
        if op_crystal[0]:
            aspect[1] = op_crystal[0]
        if op_crystal[1]:
            aspect[2] = op_crystal[1]

        # use mt_bert seq result
        if aspect[2] == -1:
            aspect[2] = op_mtbert["seq_polarity"]
        
        out.aspects = [aspect]

        return out

class CadenceMultiResolver(CadetResolverMixin):
    def __init__(self):
        pass
    
    def resolve(self, out: CadenceOutput):  
        print("[WARN] CadenceMultiResolver is not implemented")
        return out

class CadenceBertOnlyResolver(CadetResolverMixin):
    def __init__(self):
        pass
    
    def resolve(self, out: CadenceOutput):  
        print("[WARN] CadenceBertOnlyResolver is not implemented")
        return out