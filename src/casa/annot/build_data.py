from .annot_types import *
from typing import Dict, List
from itertools import chain

def build_aspect(spans: List[TextSpan], data, relevance, memo):    
    aspect = AnnotAspect()
    aspect.relevance = relevance
    aspect.memo = memo
    aspect.thread_idx = data["thread_idx"]
    aspect.batch_idx = data["batch_idx"]
    aspect.serial = data["serial"]
    aspect.spans = spans
    
    return aspect

def process_thread_annotations(annot_item, debug=False):
    comp = annot_item["completions"]
    result = comp[0]["result"]
    raw_data = annot_item["data"]

    text_spans: Dict[str, TextSpan] = {}
    annot_spans: Dict[str, List[SpanAnnot]] = {}
    relevance = ""
    memo = ""

    # Process relations
    relations = AspectRelations()
    for res_x in result:  
        if res_x["type"] != "relation":
            continue
        relations.add((res_x["from_id"], res_x["to_id"]))    

    # Prepare all annotation spans
    for res_x in result:    
        if res_x["type"] == "relation":
            continue        
        
        if res_x["from_name"] == "Relevance":
            relevance = res_x["value"]["choices"][0]
        elif res_x["from_name"] == "memo":
            memo = "\n".join(res_x["value"]["text"])
        elif res_x["type"] == "hypertextlabels":
            text_span = TextSpan.from_dict(res_x)
            text_spans[text_span.annot_id] = text_span    
        else:            
            annot_span = SpanAnnot.from_dict(res_x)                        
            span_id = annot_span.span_id
            annot_spans.setdefault(span_id, []).append(annot_span)                        

    # Link corresponding AnnotSpan and TextSpan
    for span_x in text_spans.values():
        span_id = span_x.annot_id        
        span_x.annotations = annot_spans.get(span_id, [])
    
    # Apply default relation heuristics
    link_default_relations(relations, text_spans)

    # Constructing aspects from grouping related spans 
    buf = text_spans.copy()
    aspect_list = []
    span_groups = []
    while buf:
        x = buf.pop(list(buf.keys())[0])
        neighs = relations.get(x.annot_id)
        spans = [x]    
        while neighs:
            spans.extend([buf.pop(y) for y in neighs if y in buf])
            neighs = list(chain(*[relations.get(y) for y in neighs]))    
            neighs = [x for x in neighs if x in buf]                
        
        # skip singleton span groups
        if (len(spans) > 1 or 
            spans[0].annot_type == AspectEnum.Context):            
            aspect = build_aspect(spans, raw_data, relevance, memo)
            span_groups.append(spans)
            aspect_list.append(aspect)
    
    if debug:
        return aspect_list, dict(
            relations=relations.relations, annot_spans=annot_spans,
            text_spans=text_spans, span_groups=span_groups)
    else:
        return aspect_list

def link_default_relations(relations: AspectRelations, text_spans: Dict[str, TextSpan]):
    # only apply default relations when no relations present
    if len(relations.relations) > 0:
        return
    
    # apply when there are only three spans (E, A, V)
    if len(text_spans) == 3:        
        flag = 0    
        for span_x in text_spans.values():
            annot_type = span_x.annot_type
            if annot_type == AspectEnum.Entity:
                flag |= 1
            elif annot_type == AspectEnum.Attribute:
                flag |= 2
            elif annot_type == AspectEnum.Evaluation:
                flag |= 4
        if flag == 7:
            keys = list(text_spans.keys())
            relations.add((keys[0], keys[1]))
            relations.add((keys[0], keys[2]))