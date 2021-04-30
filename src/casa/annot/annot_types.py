from enum import Enum, auto
from typing import Union, List
NormCategory = str
Rating = int

class AspectRelations:
    def __init__(self):
        self.relations = {}
    
    def add(self, rel):
        x, y = rel
        rels = self.relations
        rels.setdefault(x, []).append(y)
        rels.setdefault(y, []).append(x)
    
    def get(self, x):
        return self.relations.get(x, [])            

class AspectEnum(Enum):
    Entity = auto()
    Attribute = auto()
    Evaluation = auto()
    Context = auto()

class FieldEnum(Enum):
    NormEnt = auto()
    NormAttr = auto()
    Rating = auto()    
    RawText = auto()
    Note = auto()

    @classmethod
    def from_name(cls, field):
        field = field.lower()
        if "sentiment" in field:
            return FieldEnum.Rating
        elif "ent" in field:
            return FieldEnum.NormEnt
        elif "attr" in field:
            return FieldEnum.NormAttr
        elif "note" in field:
            return FieldEnum.Note

class SpanAnnot:
    def __init__(self):
        self.span_id = ""        
        self.from_name: str = ""
        self.payload: Union[NormCategory, Rating, str] = ""        

    def __repr__(self):
        return "<AnnotSpan [{span_id}]: {from_name}, {payload}>".format(
            span_id = self.span_id, 
            from_name = self.from_name,
            payload = self.payload
        )

    @classmethod
    def from_dict(cls, data):
        obj = SpanAnnot()
        values = data["value"]
        obj.span_id = data["id"]
        from_name = data["from_name"].lower()
        obj.from_name = from_name
        
        if "rating" in values:
            obj.payload = values["rating"]
        elif "choices" in values:
            obj.payload = ",".join(values["choices"])
        elif "text" in values:
            obj.payload = "\n".join(values["text"])
        else:
            obj.payload = ""
        return obj
    
    
class TextSpan:
    def __init__(self):
        self.annot_id = ""
        self.annot_type = None
        self.annotations: List[SpanAnnot] = []
        self.start = ""
        self.end = ""
        self.start_offset = -1
        self.end_offset = -1
        self.text = ""
        self.meta = ""

    def __repr__(self):
        return "<TextSpan [{annot_id}]: {annot_type}, {text}>".format(
            annot_id = self.annot_id, 
            annot_type = self.annot_type.name,
            text = self.text
        )

    def get_annot_value(self, field: FieldEnum):
        if field == FieldEnum.RawText:
            return self.text
        else:
            for annot_x in self.annotations:
                if FieldEnum.from_name(annot_x.from_name) == field:
                    return annot_x.payload
        return ""

    @classmethod
    def from_dict(cls, data):
        obj = TextSpan()
        obj.annot_id = data["id"]
        values = data["value"]
        obj.start = values["start"]
        obj.end = values["end"]
        obj.start_offset = values["startOffset"]
        obj.end_offset = values["endOffset"]
        obj.text = values["text"]
        obj.meta = "\n".join(data.get("meta", {}).get("text", []))        
        html_label = values["htmllabels"][0]
        obj.annot_type = AspectEnum[html_label]

        return obj


class AnnotAspect:
    def __init__(self):
        self.thread_idx = -1
        # 0: main_text, 1 and above: reply number
        self.cursor = -1
        self.batch_idx: int = -1
        self.serial: int = -1
        self.relevance: bool = False        
        self.memo: str = ""
        self.spans: List[TextSpan] = []        

    @property
    def has_context(self):
        return any(x.annot_type==AspectEnum.Context for x in self.spans)

    @property
    def has_context_only(self):
        return (any(x.annot_type==AspectEnum.Context for x in self.spans)
            and all(x.annot_type!=AspectEnum.Evaluation for x in self.spans))

    def normalized(self, aspect:AspectEnum, field: FieldEnum) -> str:
        span = self.get_aspect_span(aspect)
        if span:
            return span[0].get_annot_value(field)
        else:
            return ""

    def polarity(self) -> int:
        span = self.get_aspect_span(AspectEnum.Evaluation)
        if span:
            return span[0].get_annot_value(FieldEnum.Rating)
        else:
            return -1

    def raw_text(self, aspect) -> str:
        span = self.get_aspect_span(aspect)
        if span:
            return span[0].get_annot_value(FieldEnum.RawText)
        else:
            return ""
        
    def get_aspect_span(self, aspect: AspectEnum) -> List[TextSpan]:
        span = [x for x in self.spans if x.annot_type==aspect]
        if not span:
            span = [x for x in self.spans if x.annot_type==AspectEnum.Context]
        return span

    def make_tuple(self, use_context=False):
        E = self.normalized(AspectEnum.Entity, FieldEnum.NormEnt)
        A = self.normalized(AspectEnum.Attribute, FieldEnum.NormAttr)
        P = self.polarity()
        if use_context and self.has_context:
            V = self.raw_text(AspectEnum.Context)
        else:
            V = self.raw_text(AspectEnum.Evaluation)
        return (E, A, V, P)
        

    def __repr__(self):
        if self.has_context:
            fstr = "<AnnotAspect (ctx)"
        else:
            fstr = "<AnnotAspect (std)"
        fstr += ", {b}-{s}, Thread {t} :{E}/{A}/{V}/{P}>"
                
        E, A, V, P = self.make_tuple(use_context=False)

        return fstr.format(
            b = self.batch_idx, 
            t = self.thread_idx,
            s = self.serial,
            E = E, A = A,
            V = V, P = P
        )
    
