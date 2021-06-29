'''
ABSA: Bert aspect extraction test runner 
input: a csv file containing 'text' column and 'id' column 
output: a csv file of the input file with appended spans column, a logits pkl and true_predictions pkl
Usage: 
python bert_test.py --test_file='/content/drive/MyDrive/指向情緒案/data/threads/cht2021-JanMay-op-every20-text.csv'\
                  --pretrained_path='/content/drive/MyDrive/指向情緒案/data/models/Bert_aspect_extraction/0523'\
                  --save_dir='./Results'\ 
                  --outfile_name='out.csv' 
'''

import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, AutoModel, BertForTokenClassification
import pickle
import pandas as pd 
import argparse 
import logging
import time
import more_itertools as mit

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Run test with Bert aspect extraction")
    parser.add_argument(
        "--test_file", type=str, help="A csv file containing the testing data, containing 'text' column and 'id' column"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="Where to load the pretrained model",
        default='/content/drive/MyDrive/指向情緒案/data/models/Bert_aspect_extraction/0523'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Base model name of the pretrained model",
        default='bert-base-chinese'
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='bert-base-chinese',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--save_dir", 
        type=str, 
        default='./Results', 
        help="Where to storethe output results."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="The (ID of) the device you would like to use.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=100,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--outfile_name",
        default='out.csv',
    )
    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    return args

def translate(t):
    if t[-1] == 'P':
        return 'Positive'
    return 'Negative'

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    df = pd.read_csv(args.test_file)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    model = BertForTokenClassification.from_pretrained(args.pretrained_path, num_labels = 5)
    Original_data_size = len(df)
    df = df[df['text'].apply(lambda x: isinstance(x, str))]
    df = df.reset_index(drop = True)
    logger.info("***** Running testing *****")
    logger.info(f"  Task Name = Eval Aspect Extraction")
    logger.info(f"  Original Test Data Size = {Original_data_size}")
    logger.info(f"  Valid Test Data Size = {len(df)}")

    '''Preparing testloader'''
    # split in Chinese mixed with English can be tricky, so split beforehand
    texts = [list(x) for x in df['text']]
    tokenized_examples = tokenizer(
        texts, 
        max_length = 512,
        padding = "max_length",
        truncation = True,
        is_split_into_words = True)

    tok_sents = torch.tensor(tokenized_examples['input_ids'])
    tok_masks = torch.tensor(tokenized_examples['attention_mask'])
    test_set = TensorDataset(tok_sents, tok_masks)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)

    logger.info(f"  Evaluating Test Set ...")
    
    model.to(args.device)
    start = time.time()
    model.eval()
    test= {'preds':[], 'input_ids':[], 'logits':[]}
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch[0].to(args.device) # input_ids
            attention_mask = batch[1].to(args.device)
            
            # active_loss = attention_mask.view(-1) == 1
            logits = model(input_ids = input_ids, attention_mask=attention_mask).logits
            # active_logits = logits.view(-1, 5) # num of labels = 5
            
            _, pred = torch.max(logits, 2)
            test['preds'].extend(pred.cpu().numpy())
            test['input_ids'].extend(input_ids.cpu().numpy())
            test['logits'].extend(logits.cpu().numpy())
    end = time.time()
    logger.info(f"  Total Inference Time: {(end - start)//60} min {round((end-start)%60, 4)} s")
    
    '''Cleaning Predictions'''
    logger.info(f"  Extracting Spans...")
    label_list = {0: 'B-VN', 1:'B-VP',  2:'I-VN', 3: 'I-VP', 4:'O'}
    special_tokens = [101, 102] # 不應該拔UNK，只拔101, 102
    predictions_all = test['preds']
    inputids_all = test['input_ids']

    ### !!! 把predictions的第一個token（CLS）跳掉，然後一直讀到倒數第二個（因為最後一個是SEP），然後把這個predictions和原始tokens zip起來
    true_predictions = [[label_list[p] for (p, inpid) in zip(predsent, inpidsent) if inpid not in special_tokens] 
                        for predsent, inpidsent in zip(predictions_all, inputids_all)]
    
    # find the labeled span 
    eval_spans = {}
    text_idxes = df['id'].tolist()
    assert len(texts) == len(true_predictions)
    for idx, text, pred in zip(text_idxes, texts, true_predictions):
        pred_spans_for_one_text = []
        pred = pred[:len(text)]
        pairs = [(i, val) for i, val in enumerate(pred) if val != 'O']
        span_groups = []
        if len(pairs) >1:
            polarity = list(map(lambda x: x[1], pairs))
            idxes = [x for (x, val) in pairs]
            idxes_groups = [list(group) for group in mit.consecutive_groups(idxes)]
            for group in idxes_groups:
                label = ''.join([text[i] for i in group])
                if len(label)>0: 
                    span_text = label
                    span_pol = translate(pred[group[0]])  # assume each span to have the same polarity
                    span_groups.append((span_pol, span_text))
            # print(span_groups)
        # adding back to span dictionary 
        if idx not in eval_spans:
            eval_spans[idx] = []
            eval_spans[idx].extend(span_groups)
        else: eval_spans[idx].extend(span_groups)    
    
    '''Outputting file'''
    spans_series = []
    for idx in text_idxes: 
        # print(idx)
        spans_series.append(eval_spans[idx])
    spans_series = pd.Series(spans_series, name="pred_eval_spans")
    df['pred_eval_spans'] = spans_series
    df.to_csv(os.path.join(args.save_dir, args.outfile_name))

    with open(os.path.join(args.save_dir, "preds.pkl"), 'wb') as f:
        pickle.dump(true_predictions, f)
    with open(os.path.join(args.save_dir, "logits.pkl"), 'wb') as f:
        pickle.dump(test['logits'], f)
    logger.info(f"  Files saved at {args.save_dir}.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
