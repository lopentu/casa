from .opinion_types import Opinion

class OpinionProc(Opinion):    
    def __init__(self):
        super(OpinionProc, self).__init__()
        self.proc_info = {}
        self.parse_tree = None
        self.tokens = None        

    @classmethod
    def from_opinion(self, opinion:Opinion):
        opp = OpinionProc()
        opp.__dict__.update(opinion.__dict__)
        return opp
    
    

    