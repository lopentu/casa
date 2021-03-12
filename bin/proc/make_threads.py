from argparse import ArgumentParser
from import_casa import casa
import logging
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def main(args):
    csv_data = pd.read_csv(args.data_path)    
    csv_data = csv_data.replace(np.nan, '')    
    data_iter = (row.to_dict() for _, row in csv_data.iterrows())
    threads = casa.make_opinion_threads(data_iter)
    n_opinion = sum([len(x) for x in threads])
    logging.info("%d threads built from data", len(threads))
    logging.info("%d opinions in threads", n_opinion)
    
    with open(args.out_path, "wb") as fout:
        pickle.dump(threads, fout)
    logging.info("pickle data into %s", args.out_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", help="the csv data path")
    parser.add_argument("--out-path", help="the output pickle path")    

    args = parser.parse_args()
    if not args.out_path:
        args.out_path = args.data_path.replace(".csv", ".pkl")
    main(args)


