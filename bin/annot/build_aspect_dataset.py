import json
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from import_casa import casa, cano  #type: ignore

logging.basicConfig(level="INFO")

def build_proofread(json_paths, out_path):
    rec_list = []    
    for json_path in tqdm(json_paths):
        with json_path.open("r", encoding="UTF-8") as fin:
            annots = json.load(fin)
        for annot_i, annot_x in enumerate(annots): 
            try:
                aspects = cano.process_thread_annotations(annot_x)
                records = cano.make_proofread(aspects, annot_x["data"]["html"])
                rec_list.extend(records)
            except:
                pass
    proof_df = pd.DataFrame.from_dict(rec_list)
    proof_df = proof_df.loc[proof_df.norm_val.str.len() > 0, :]
    proof_df.reset_index(inplace=True)    
    proof_df.to_csv(out_path)

def build_seqs(json_paths, out_path):
    seq_pairs_list = []
    noise_pairs_list = []
    for json_path in tqdm(json_paths):
        with json_path.open("r", encoding="UTF-8") as fin:
            annots = json.load(fin)
        for annot_i, annot_x in enumerate(annots): 
            try:
                aspects = cano.process_thread_annotations(annot_x)
                seq_pairs, noise_pairs = cano.make_sequence_from_aspects(aspects, annot_x["data"]["html"], noise_ratio=0.5)
                seq_pairs_list.extend(seq_pairs)
                noise_pairs_list.extend(noise_pairs)
            except Exception:
                logging.warning("Sequence exception in ", annot_i)

    with open(out_path, "wb") as fout:
        pickle.dump(seq_pairs_list, fout)
    with open(str(out_path).replace("seq", "noise"), "wb") as fout:
        pickle.dump(noise_pairs_list, fout)       
                        

def build_aspect_tuples(json_paths, out_path):
    aspect_list = []
    for json_path in tqdm(json_paths):
        with json_path.open("r", encoding="UTF-8") as fin:
            annots = json.load(fin)
        for annot_i, annot_x in enumerate(annots):        
            aspects = cano.process_thread_annotations(annot_x)        
            aspect_list.extend(aspects)
    
    
    data_items = []
    for aspect_x in aspect_list:
        batch_idx = aspect_x.batch_idx
        thread_idx = aspect_x.thread_idx
        serial = aspect_x.serial
        aspect_tuple = aspect_x.make_tuple()
        memo = aspect_x.memo
        ent_rawtext = aspect_x.raw_text(cano.AspectEnum.Entity)
        attr_rawtext = aspect_x.raw_text(cano.AspectEnum.Attribute)
        is_context = aspect_x.has_context_only
        if all(not x.strip() for x in aspect_tuple[0:3]):
            continue
        data_items.append((batch_idx, serial, thread_idx, is_context,
                        *aspect_tuple, ent_rawtext, attr_rawtext))
    
    aspect_df = pd.DataFrame(data_items, 
             columns=["batch_idx", "serial", "thread_idx", "is_context", "ent_norm", "attr_norm", "evaltext", 
                      "rating", "ent_rawtext", "attr_rawtext"])
    aspect_df.to_csv(out_path, encoding="UTF-8", index=False)
    logging.info("aspect tuples is written to %s", out_path)


def main(args):
    
    annot_data_dir = casa.get_data_path() / "annot_data/annotated_data_bkup"
    if args.target_dir:
        target_dir = args.target_dir
    else:        
        chdirs = sorted([x for x in annot_data_dir.iterdir() if x.is_dir])
        target_dir = chdirs[-1].name

    result_dir = casa.get_data_path() / f"annot_data/annotated_data_bkup/{target_dir}"
    aspect_tuple_path = result_dir/f"aspect_tuples_{target_dir}.csv"
    proofread_path = result_dir / f"aspects_proof_read_{target_dir}.csv"
    seqs_path = result_dir / f"seq_pairs_{target_dir}.pkl"
    json_paths = [x for x in result_dir.iterdir() if x.suffix==".json"]
    
    build_aspect_tuples(json_paths, aspect_tuple_path)
    build_seqs(json_paths, seqs_path)
    build_proofread(json_paths, proofread_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--target_dir", default="", type=str)
    args = parser.parse_args()
    
    main(args)