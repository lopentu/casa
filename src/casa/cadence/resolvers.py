
from enum import Enum, auto

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

class CadenceSimpleResolver:
    def __init__(self, output: CadenceOutput):
        self.output = output
    
    def resolve(self):
        out = self.output        
        op_crystal = out.crystal["result"]
        op_cadet = out.cadet    
        op_mtbert = out.mt_bert
        aspect = [op_cadet["entity"][0], op_cadet["service"][0], -1]

        # use crystal result if available
        if op_crystal[0]:
            aspect[1] = op_crystal[0]
        if op_crystal[1]:
            aspect[2] = op_crystal[1]

        # use mt_bert seq result
        if aspect[2] == -1:
            aspect[2] = op_mtbert["seq_polarity"]
        
        self.output.aspects = [aspect]

        return self.output

class CadenceMultiResolver:
    def __init__(self, output: CadenceOutput):
        self.output = output
    
    def resolve(self):
        print("[WARN] CadenceMultiResolver is not implemented")
        return self.output

class CadenceBertOnlyResolver:
    def __init__(self, output: CadenceOutput):
        self.output = output
    
    def resolve(self):
        print("[WARN] CadenceBertOnlyResolver is not implemented")
        return self.output