import re
from itertools import chain
from .annot_types import AspectEnum
from lxml import objectify
from lxml.html import html5parser

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

def make_sequence_from_aspects(aspects, html_text):
    parent_div = make_div_element(html_text)
    cursor_spans_pairs = collect_sentences(chain(*(x.spans for x in aspects)))    
    seq_pairs = []
    for cursor, spans in cursor_spans_pairs.items():
        pairs = mark_tokens(cursor, spans, parent_div)
        seq_pairs.extend(pairs)
    return seq_pairs

def print_seq_pair(pair):
    print(" ".join(f"{x}({t})" for x, t in zip(*pair)))