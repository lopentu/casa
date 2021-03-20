from argparse import ArgumentParser
import argparse
import json
import re
import logging
from pathlib import Path
from tqdm.auto import tqdm
from import_casa import casa
import pandas as pd

logging.basicConfig(level="INFO")

def main(args):
    dir_path = casa.get_data_path() / "annot_data/annotated_data_bkup"
    
    if args.target_dir:
        target_name = args.dir_name
    else:        
        chdirs = sorted([x for x in dir_path.iterdir() if x.is_dir])
        target_name = chdirs[-1].name    
    
    target_dir = dir_path / target_name
    annot_paths = [x for x in target_dir.iterdir() if x.suffix==".json"]

    annots = []        
    for annot_file in tqdm(annot_paths):
        with open(annot_file, "r", encoding="UTF-8") as fin:
            annot_data = json.load(fin)
            try:
                src_label = re.search("\d+", annot_file.name).group(0)
            except:
                src_label = "NA"
            dframe = casa.annot.convert_annot_result(annot_data, src_label)
            annots.append(dframe)
    mframe = pd.concat(annots)
    mpath = target_dir / f"merged_frame_{target_name}.csv"
    mframe.to_csv(mpath)
    logging.info("Merged dataframe is written to " + str(mpath))

parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", default="", type=str)
args = parser.parse_args()
if __name__ == "__main__":
    main(args)