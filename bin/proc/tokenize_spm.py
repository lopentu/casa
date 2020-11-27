from argparse import ArgumentParser
from import_casa import casa
import logging
import numpy as np
import pandas as pd
import pickle

def main(args):
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pkl_path", help="the pickle path")
    parser.add_argument("model_prefix", help="the output pickle path")    

    args = parser.parse_args()    
    main(args)