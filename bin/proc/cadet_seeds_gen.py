from argparse import ArgumentParser
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import logging
logging.basicConfig(level="INFO")

def main(args):
    seed_list = pd.read_csv(args.seed_path)        
    assert "Seed" in seed_list.columns.values
    candidates = []
    kv = KeyedVectors.load(str(args.ft_path))

    for row in seed_list.itertuples():
        seed = row.Seed
        rets = kv.most_similar(str(seed.lower()), topn=20)
        candids = [x[0] for x in rets]
        if seed in kv.vocab:
            candids.insert(0, seed)
        candidates.append(candids)
    seed_list["candidates"] = [",".join(candid_x) for candid_x in candidates]
    candid_path = args.seed_path.replace(".csv", ".candids.csv")
    seed_list.to_csv(candid_path, encoding="utf8", errors='ignore')
    logging.info("Seeded list is written to %s", candid_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("seed_path")
    parser.add_argument("ft_path")

    args = parser.parse_args()
    main(args)