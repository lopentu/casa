import re
from itertools import chain
from .annot_types import (AnnotAspect, AspectEnum, 
                          SpanAnnot, TextSpan, FieldEnum)
from typing import List
from lxml import objectify
from lxml.html import html5parser
import random

def extract_sentence_cursor(xpath):
    cursor = xpath.replace("/div[1]/", "")
    cursor = re.sub(r"/span\[\d+\]", "", cursor)
    cursor = cursor[:cursor.find("/text()")+7]
    return cursor

def collect_sentences(spans):    
    sentence_spans = {}    
    for span_x in spans:
        start_cursor = extract_sentence_cursor(span_x.start)
        end_cursor = extract_sentence_cursor(span_x.end)
        if start_cursor != end_cursor:
            # print("cross span: ", start_cursor, end_cursor)
            pass
        sentence_spans.setdefault(start_cursor, [])\
            .append(span_x)
    return sentence_spans

def fill_tag(start, end, annot_type, tags):
    tag_cat = {
        AspectEnum.Entity: "E",
        AspectEnum.Attribute: "A",
        AspectEnum.Evaluation: "V",
        AspectEnum.Context: "C",
    }.get(annot_type, "O")
    
    for i in range(start, end):
        prefix = "B-" if i==start else "I-"
        tags[i] = prefix+tag_cat

def clean_space(text):    
    text = text.replace("\n", ' ')
    return re.sub("(?<=[^a-zA-Z0-9])\s+", "", text.strip())

def split_long_sequence(text, tags):
    text_list = []
    tags_list = []
    assert len(text) == len(tags)
    offset = 0
    for match in re.finditer("[。？！?!)]+", text):
        mlen = match.end() - match.start()
        text_list.append(text[offset:match.start()+mlen])
        tags_list.append(tags[offset:match.start()+mlen])
        offset = match.start()+mlen
    if offset < len(text):
        text_list.append(text[offset:])
        tags_list.append(tags[offset:])
    return text_list, tags_list
    
def mark_tokens(cursor, spans, target_elem):
    rawtext = target_elem.xpath(cursor)    
    rawtext = rawtext[0] if rawtext else ""
    rawtext = clean_space(rawtext)    
    tags = ["B-O"] * len(rawtext)
    span_history = {}
    
    for span_x in spans:
        spantext = clean_space(span_x.text) 
        count_x = rawtext.count(spantext)
        if count_x == 0:
            # only check for the simple case, where it is a reply
            if span_x.start != span_x.end:
                print("different text spans: ", spantext)
                continue

            print("warning, text not found: ", rawtext, ">>", spantext) 
            breakpoint()           
            raise ValueError()            
        else:
            idx = rawtext.find(spantext, span_history.get(spantext, 0))
            if idx < 0:
                breakpoint()

            span_history[spantext] = idx            
            fill_tag(idx, idx+len(spantext), span_x.annot_type, tags)
    
    if len(rawtext) < 300:        
        text_list, tags_list = [rawtext], [tags]
    else:
        text_list, tags_list = split_long_sequence(rawtext, tags)
    
    pairs = [(text, tags) for text, tags
             in zip(text_list, tags_list)
             if len(tags) > 5]
    return pairs

def make_div_element(html_str):
    div = html5parser.fragment_fromstring(html_str)
    for elem_x in div.getiterator():
        idx = elem_x.tag.find('}')
        elem_x.tag = elem_x.tag[idx+1:]
    objectify.deannotate(div, cleanup_namespaces=True)
    return div

def generate_cursors(parent_div):
    lis = parent_div.xpath("//ol/li")
    cursors = []
    for li_idx in range(len(lis)):
        cursors.append(f"ol[1]/li[{li_idx+1}]/text()")
    return cursors

def generate_noise_seq(cursor, target_elem):
    rawtext = target_elem.xpath(cursor)        
    rawtext = rawtext[0] if rawtext else ""
    rawtext = clean_space(rawtext)    
    rawtext = rawtext[:300]
    tags = ["B-O"] * len(rawtext)
    return (rawtext, tags)

def build_proofread_entry(aspect_x: AnnotAspect, span_x: TextSpan, field, cursor, rawtext):
    rec = dict(
                annot_type=span_x.annot_type.name,
                norm_val=span_x.get_annot_value(field),
                spantext=span_x.text,
                rawtext=rawtext,
                batch_idx=aspect_x.batch_idx,
                thread_idx=aspect_x.thread_idx,
                serial=aspect_x.serial,
                cursor=cursor,
                span_id=span_x.annot_id
            )
    return rec

def make_proofread(aspects: List[AnnotAspect], html_text: str):
    parent_div = make_div_element(html_text)    
    work_buf = {}
    data_list = []

    for aspect_x in aspects:
        # make attribute and entity spans for proofread only
        entity_span = aspect_x.get_aspect_span(AspectEnum.Entity)
        attrib_span = aspect_x.get_aspect_span(AspectEnum.Attribute)
        spans = entity_span + attrib_span
        
        for span_x in spans:
            cursor = extract_sentence_cursor(span_x.start)
            rawtext = parent_div.xpath(cursor)    
            rawtext = rawtext[0] if rawtext else ""
            rawtext = clean_space(rawtext)
            
            # remove the same textspan annotated in the same text
            if cursor not in work_buf:
                work_buf[cursor] = rawtext

            if span_x.text in work_buf[cursor]:
                work_buf[cursor] = work_buf[cursor].replace(span_x.text, "", 1)
            else:
                continue

            # There is a new instance in the text, continue making a span record
            annot_type = span_x.annot_type
            if annot_type in (AspectEnum.Attribute, AspectEnum.Context):
                data_list.append(
                    build_proofread_entry(aspect_x, span_x, FieldEnum.NormAttr,
                                    cursor, rawtext))
            elif annot_type in (AspectEnum.Entity, AspectEnum.Context):
                data_list.append(
                    build_proofread_entry(aspect_x, span_x, FieldEnum.NormEnt,
                                    cursor, rawtext))


    return data_list

def make_sequence_for_caprice(aspects, html_text, noise_ratio=0.5):
    parent_div = make_div_element(html_text)
    cursor_spans_pairs = collect_sentences(chain(*(x.spans for x in aspects)))        
    seq_pairs = []
    seq_labels = []
    for cursor, spans in cursor_spans_pairs.items():
        pairs = mark_tokens(cursor, spans, parent_div)
        if len(pairs) != 1: # skip long text
            continue        

        labels = [x.get_annot_value(FieldEnum.Rating) for x in spans]
        labels = [x for x in labels if x]
        if (labels and
            all(x==labels[0] for x in labels) and
            labels[0] != 3):
            # only include those sequence having appropriate labels
            rating = int(labels[0])
            assert len(pairs) == 1
            seq_pairs.append(pairs[0])            
            seq_labels.append(int(rating>3)+1)
        else:
            continue

    cursor_pools = generate_cursors(parent_div)                

    for cursor in cursor_pools:
        if cursor in cursor_spans_pairs:
            continue
                    
        if random.random() > noise_ratio:
            continue
        
        noise_pair = generate_noise_seq(cursor, parent_div)
        seq_pairs.append(noise_pair)
        seq_labels.append(0)
        
    return seq_pairs, seq_labels

def make_sequence_from_aspects(aspects, html_text, noise_ratio=0.0):
    parent_div = make_div_element(html_text)
    cursor_spans_pairs = collect_sentences(chain(*(x.spans for x in aspects)))        
    seq_pairs = []
    
    for cursor, spans in cursor_spans_pairs.items():
        pairs = mark_tokens(cursor, spans, parent_div)
        seq_pairs.extend(pairs)

    if noise_ratio == 0.0:    
        return seq_pairs

    else:
        cursor_pools = generate_cursors(parent_div)        
        noise_pairs = []        

        for cursor in cursor_pools:
            if cursor in cursor_spans_pairs:
                continue
                        
            if random.random() > noise_ratio:
                continue
            
            noise_pair = generate_noise_seq(cursor, parent_div)
            noise_pairs.append(noise_pair)
            
        return seq_pairs, noise_pairs
    

def print_seq_pair(pair):
    print(" ".join(f"{x}({t})" for x, t in zip(*pair)))