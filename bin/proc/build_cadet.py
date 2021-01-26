from argparse import ArgumentParser
import re
import logging
from pathlib import Path
import pickle
from tqdm.auto import tqdm
from import_casa import casa

logging.basicConfig(level="INFO")

def build_fastText(token_path, ft_path):
    logging.info("Build SP-FastText")
    from gensim.models import FastText
    from gensim.models.word2vec import LineSentence
    sp_sentences = LineSentence(token_path)
    model = FastText(size=200, window=5, min_count=3)
    model.build_vocab(sentences=sp_sentences)
    model.train(sentences=sp_sentences, 
                total_examples=model.corpus_count, epochs=5)
    model.wv.save(str(ft_path))

def build_spm_tokens(text_path, token_path, sp_model_path):
    logging.info("Build SPM tokens")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    fin = open(text_path, "r", encoding="UTF-8")
    fout = open(token_path, "w", encoding="UTF-8")
    pat = re.compile("[,▁]")
    for ln in tqdm(fin):
        tokens = sp.encode(ln.strip(), out_type=str)
        preproc_iter = map(lambda x: pat.sub("", x), tokens)
        fout.write(" ".join(preproc_iter) + "\n")

    fin.close()
    fout.close()

def build_spm(text_path, sp_path):
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
            input=text_path, vocab_size=10000,
            model_prefix=sp_path,
            model_type="bpe",
            add_dummy_prefix=False,
            split_by_unicode_script=False,
            split_by_number=False)


def build_text(pkl_path, text_path):
    with open(pkl_path, "rb") as fin:
        threads = pickle.load(fin)

    logging.info("Building plain text from threads")
    fout = open(text_path, "w", encoding="UTF-8")
    for thread_x in tqdm(threads):
        for ln in thread_x.opinion_texts():
            ln = ln.lower()
            fout.write(ln.strip()+"\n")
    fout.close()

def main(args):
    data_dir = casa.get_data_path()
    base_dir = data_dir / args.prefix
    base_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = args.pkl_path
    txt_path = Path(pkl_path.replace(".pkl", ".txt"))
    token_path = Path(pkl_path.replace(".pkl", ".spm.txt"))
    sp_path = base_dir / "spm-2020"
    ft_path = base_dir / "ft-2020.kv"
    sp_model_path = Path(str(sp_path) + ".model")

    if not txt_path.exists() or args.update:
        build_text(pkl_path, txt_path)

    if not sp_model_path.exists() or args.update:
        build_spm(txt_path, sp_path)

    if not token_path.exists() or args.update:
        build_spm_tokens(txt_path, token_path, sp_model_path)

    if not ft_path.exists() or args.update:
        build_fastText(token_path, ft_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pkl_path", help="the pickle path")
    parser.add_argument("--prefix", help="the output pickle path", default="cadet/op20")
    parser.add_argument("--update", help="ignore existing model files", action="store_true")

    args = parser.parse_args()
    main(args)