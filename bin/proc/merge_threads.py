import glob
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from import_casa import casa

def main(args):    
    infiles = glob.glob(args.pickle_files)
    threads = []
    print("Merging threads...")
    for infile_x in tqdm(infiles):        
        fin = open(infile_x, "rb")
        indat = pickle.load(fin)
        threads.extend(indat)
        fin.close()
    with open(args.output_path, "wb") as fout:
        pickle.dump(threads, fout)
    print("Merged threads: ", len(threads))
    print("Merged threads are written to", args.output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pickle_files")
    parser.add_argument("output_path")
    args = parser.parse_args()
    main(args)