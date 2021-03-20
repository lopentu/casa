import pandas as pd

def convert_annot_result(annot_obj, src_label):
    colnames = "src,thread_idx,serial,from_name,ent_type,ent_choice,ent_text".split(",")
    entries = []
    for annot_x in annot_obj:
        entries.extend(extract_result(annot_x, src_label))
    return pd.DataFrame(entries, columns=colnames)

def extract_result(annot_item, src_label):
    serial = annot_item["data"]["serial"]
    thread_idx = annot_item["data"]["thread_idx"]
    result = annot_item["completions"][0]["result"]
    
    entries = []
    for x in result:
        from_name = x.get("from_name", "")
        ent_type = x["type"]
        value_dict = x.get("value", {})
        ent_choice = value_dict.get("choices", [""])[0]
        ent_rating = value_dict.get("rating", -1)
        ent_text = value_dict.get("text", "")
        ent_value = ent_rating if ent_type == "rating" else ent_choice
        if isinstance(ent_text, list):
            ent_text = "".join(ent_text)
        entries.append((
            src_label,
            thread_idx, serial,
            from_name, ent_type, ent_value, ent_text
        ))
        
    return entries