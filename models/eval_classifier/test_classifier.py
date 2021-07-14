'''
ABSA: BERT sentiment classifer Test Runner
Usage (eg) (colab-style)
!python3 test_classifier.py --test_file /content/drive/MyDrive/指向情緒案/data/annot_data/annotated_data_bkup/20210707/aspect_tuples_20210707.csv\
        --model_path /content/drive/MyDrive/指向情緒案/data/models/BERT_eval_classifier/0708_0.759ctx\
        --model_name 'bert-base-chinese'\
        --MAX_LEN 400\
        --eval_batch_size 100\
        --save_dir ./test_results

2021.7.8
'''
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, AutoModel, BertForSequenceClassification
from transformers import AdamW
import pandas as pd 
import argparse 
import logging
import numpy as np 
import pickle
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune or Run test on BERT sent classifier")
    parser.add_argument(
        "--test_file", 
        type=str, 
        help="Path to a csv file containing training data ('evaltext' column)\
            the default setting is a .csv similar to aspect_tuples.csv.\
            If 'rating' col is provided, a acc score is calculated", 
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Where to load the pretrained model",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="base model name of the model_path",
        default='bert-base-chinese'
    )

    parser.add_argument(
        "--MAX_LEN",
        help="Max length of one input data",
        type = int,
        default = 512, 
    )

    parser.add_argument(
        "--eval_batch_size",
        type =int,
        default = 32,
        help="Eval batch size",
    )

    parser.add_argument("--save_dir", 
        type = str, 
        default = './Results', 
        help="Where to store the output results and the fine-tuned model."
    )

    parser.add_argument("--outfile", 
        type = str, 
        default = "out.csv", 
        help = "The predict file name."
    )

    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    return args


# utils 
def to_pol(List):
  pols = []
  for rating in List: 
    if rating > 3: pols.append(1)
    elif rating == 3: pols.append(0)
    else: pols.append(2)
  assert len(pols) == len(List)
  return pols

def trans(List):
  new = []
  for i in List:
    if i == 1: new.append('Positive')
    elif i == 2: new.append('Negative')
    else: new.append('Neutral')
  return new

def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("***** Testing *****")
    logger.info(f"  Task Name = Sentiment classification")

    df = pd.read_csv(args.test_file)


    df = df.dropna()
    df = df.reset_index()
    # if the file comes with gold labels
    GOLD = False
    if 'polarity' in df:
        GOLD = True
    else: df= df[['evaltext']]

    # df.to_csv(os.path.join(args.save_dir, 'test_file_clean.csv'), index = False)
    
    
    LEN = len(df)
    logger.info(f"  Test size = {LEN}")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels = 3)

    MAX_LEN = args.MAX_LEN
    
    tokenized_examples = tokenizer.batch_encode_plus(
        df['evaltext'].tolist(),
        max_length = MAX_LEN,
        padding=True,
        truncation=True
    )


    seq = torch.tensor(tokenized_examples['input_ids'])
    mask = torch.tensor(tokenized_examples['attention_mask'])
    if GOLD: 
        y = torch.tensor(df['polarity'].tolist()) 
        testset = TensorDataset(seq, mask, y)
    else: testset = TensorDataset(seq, mask)
    test_loader = DataLoader(testset, batch_size = args.eval_batch_size, shuffle = False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    

    # Testing 
    model.eval()
    test_acc = 0
    PREDS, LABELS, LOGITS = [], [], []
    # LOGITS = np.zeros(shape=(LEN+1,3))
    for step, batch in enumerate(test_loader):

        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, label = batch

        with torch.no_grad():
            pred = model(sent_id, mask).logits
            _, pred2 = torch.max(pred, 1)
            # saving logits/unnormalized probabilities of 3 classes
            
            LOGITS.extend(pred.cpu().numpy())
            # saving the predicted classes 
            PREDS.extend(pred2.cpu().tolist())
            if GOLD:
                LABELS.extend(label.cpu().tolist())
                assert len(PREDS) == len(LABELS)
                test_acc += (pred2.cpu() == label.cpu()).sum().item()
    # turn all training data logits into numpy array 
    LOGITS = np.vstack(LOGITS)
    D = {'texts': df['evaltext'] , 'labels':trans(LABELS), 'preds':trans(PREDS)}
    if GOLD:
        logger.info(f'  Test acc = {(test_acc / len(df)):.3f}')
    logger.info(f'  Saving logits of shape {LOGITS.shape} to Logits.pkl...')
    with open(os.path.join(args.save_dir, 'Logits.pkl'), 'wb') as f:
        pickle.dump(LOGITS, f)
    preds = pd.DataFrame.from_dict(D)
    logger.info(f'  Saving predictions to {args.outfile}...')
    preds.to_csv(os.path.join(args.save_dir, args.outfile))
    
    logger.info(f'  Finished.')

if __name__ == "__main__":
    args = parse_args()
    main(args)