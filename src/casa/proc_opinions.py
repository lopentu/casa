from .opinion_types import Opinion

class OpinionProc(Opinion):    
    def __init__(self, data_dict={}):
        super(OpinionProc, self).__init__(data_dict)
        self.proc_info = {}
        self.parse_tree = None
        self.text_tokens = None        
        self.title_tokens = None

    @classmethod
    def from_opinion(self, opinion:Opinion):
        opp = OpinionProc()
        opp.__dict__.update(opinion.__dict__)
        return opp

    def __repr__(self):
        text = self.text[:20]+"..." if len(self.text)>20 else self.text
        tag = []
        if self.text_tokens: tag.append("tok")
        if self.parse_tree: tag.append("tree")
        return f"<OpinionProc [{self.source}] {self.author}: {text} ({'|'.join(tag)})>"
    
    

    