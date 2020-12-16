import re

# 可以/再/[pos=V.]/一點
class PatToken:
    def __init__(self, spec):
        self.name = None
        self.word = None
        self.pos = None
        self.rel = None
        self.head = None
        name = re.findall("^(\w+):", spec)
        if name:
            self.name = name[0]
            spec = spec.split(":")[1]
        kvp = re.findall(r"\[(\w+)=(.*?)\]", spec)
        if not kvp:
            self.word = spec
        else:
            for key, value in kvp:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError("not supported key: " + key)
    
    def __repr__(self):
        kvp = {k: v for k, v in self.__dict__.items() if v}
        return f"<PatToken: {str(kvp)}>"

    def match(self, token):
        if self.word and not re.match(self.word, token[0]):
            return False
        if self.pos and not re.match(self.pos, token[1]):
            return False
        if self.rel and not re.match(self.rel, token[2]):
            return False
        if self.head and not re.match(self.head, token[3]):
            return False
        return True
    
def make_pat_tokens(pattern):
    tokens = [PatToken(x) for x in pattern.split("/")]
    
    return tokens