import pickle
import numpy as np
from sklearn.decomposition import DictionaryLearning

if __name__ == "__main__":
    with open("../../data/caprice/caprice_seq_states.pkl", "rb") as fin:
        states = pickle.load(fin)

    hiddens = np.vstack([x["hiddens"][1:, :] for x in states])
    dict_learn = DictionaryLearning(
                    fit_algorithm="cd", 
                    transform_algorithm="lasso_cd", 
                    positive_code=True, verbose=True)

    codes = dict_learn.fit_transform(hiddens)
    with open("../data/caprice/caprice_dict_codes.pkl", "wb") as fout:
        pickle.dump(fout, codes)
    with open("../data/caprice/caprice_dict_learn.pkl", "wb") as fout:
        pickle.dump(fout, dict_learn)
