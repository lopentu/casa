'''
ABSA: BERT sentiment classifer Trainer
Usage:(eg.) (colab-style)
!python3 run_classifier.py --train_file /content/drive/MyDrive/指向情緒案/data/annot_data/annotated_data_bkup/20210707/aspect_tuples_20210707.csv\
        --save_dir /content/drive/MyDrive/指向情緒案/data/models/BERT_eval_classifier/0708_0.759ctx\
        --model_name 'bert-base-chinese'\
        --valid_ratio 0.2\
        --seed 2929\
        --MAX_LEN 400\
        --batch_size 100\
        --epoch 15 

2021.7.8
'''

import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import transformers
from transformers import BertTokenizerFast, AutoModel, BertForSequenceClassification
from transformers import AdamW
import pandas as pd 
import argparse 
import logging
import time
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune or Run test on BERT sent classifier")
    parser.add_argument(
        "--train_file", 
        type=str, 
        help="Path to a csv file containing training data, containing 'evaltext', 'rating', 'is_context' column\
            the default setting is a .csv similar to aspect_tuples.csv", 
    )
    
    parser.add_argument(
        "--valid_ratio", type=float, default = 0.1, help="The split ratio of train_file to train set and dev set")
    

    parser.add_argument(
        "--MAX_LEN",
        type=int,
        default = 512, 
        help="Max length of one input data",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Base model name of pretrained model, can be a fine-tuned model path",
        default='bert-base-chinese'
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default = 0,
        help="random seed", 
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int, 
        help="Training batch size",
    )

    parser.add_argument("--save_dir", 
        type=str, 
        default='./Results', 
        help="Where to store the output results and the fine-tuned model."
    )

    parser.add_argument(
        "--epoch",
        type = int,
        default= 15,
    )

    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    return args

# util functions
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

# main function
def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("***** Training *****")
    logger.info(f"  Task Name = Sentiment classification")

    # Preparing train, test dataloders
    df = pd.read_csv(args.train_file)
    df = df.dropna()
    df = df[df['is_context'] == False]
    df = df.reset_index()
    polarity = pd.Series(to_pol(df['rating']), name = 'polarity')
    concat = df['evaltext'].to_frame().join(polarity)
    concat.to_csv(os.path.join(args.save_dir, 'train_file_clean.csv'))
    
    concat['evaltext'] = concat['evaltext'].apply(lambda x: x[:args.MAX_LEN])
    X = concat['evaltext']
    y = concat['polarity']

    # Stratification on validation 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.valid_ratio, stratify=y)

    logger.info(f"  Training Data Size = {len(X_train)}")
    logger.info(f"  Validation Data Size = {len(X_test)}")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels = 3)
    
    valid_texts = X_test.tolist()
    train_texts_LEN = len(X_train)
    X_train = tokenizer.batch_encode_plus(
        X_train.tolist(),
        max_length = args.MAX_LEN,
        padding=True,
        truncation=True
    )
    X_test = tokenizer.batch_encode_plus(
        X_test.tolist(),
        max_length = args.MAX_LEN,
        padding=True,
        truncation=True
    )

    train_seq = torch.tensor(X_train['input_ids'])
    train_mask = torch.tensor(X_train['attention_mask'])
    train_y = torch.tensor(y_train.tolist(), dtype=torch.int64)

    test_seq = torch.tensor(X_test['input_ids'])
    test_mask = torch.tensor(X_test['attention_mask'])
    test_y = torch.tensor(y_test.tolist()) 
   
    batch_size = args.batch_size
    train_data = TensorDataset(train_seq, train_mask, train_y)
    test_data = TensorDataset(test_seq, test_mask, test_y)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)


    myseed = args.seed
    os.environ['PYTHONHASHSEED'] = str(myseed)
    random.seed(myseed) 
    torch.manual_seed(myseed)
    # gpu 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed) 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr = 1e-5, weight_decay = 5e-3)
    epochs = args.epoch
    loss_fn = nn.CrossEntropyLoss()

    def train(dataloader = train_loader):
        losses, acc = 0, 0
        # iterate over batches
        for step, batch in enumerate(dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, label = batch

            # clear previously calculated gradients 
            model.zero_grad()        
            pred = model(sent_id, mask).logits

            _, pred2 = torch.max(pred, 1)
            loss = loss_fn(pred, label)
            loss.backward()
            acc += (pred2.cpu() == label.cpu()).sum().item()
            losses += loss.item()  

            # clip the the gradients to 1.0. 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        avg_acc = acc / train_texts_LEN
        avg_loss = losses / len(dataloader)

        return avg_loss, avg_acc
    
    def validate(dataloader = test_loader):
        model.eval()
        val_losses, val_acc = 0, 0
        PREDS, LABELS = [], []
        for step, batch in enumerate(dataloader):

          # push the batch to gpu
          batch = [t.to(device) for t in batch]
          sent_id, mask, label = batch

          with torch.no_grad():
            pred = model(sent_id, mask).logits
            loss =loss_fn(pred,label)
            _, pred2 = torch.max(pred, 1)
            val_acc += (pred2.cpu() == label.cpu()).sum().item()
            PREDS.extend(pred2.cpu().tolist())
            LABELS.extend(label.cpu().tolist())
            assert len(PREDS) == len(LABELS)
            val_losses += loss.item()
        avg_acc = val_acc / len(valid_texts)
        avg_loss = val_losses / len(dataloader)
        
        return avg_loss, avg_acc, PREDS, LABELS
    
    
    best_acc = 0.
    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}
    start = time.time()

    # training loop 
    for epoch in range(epochs):
        logger.info(f'  [{epoch+1}/{epochs}]')
        
        # training 
        model.train()
        train_loss, train_acc = train()
        val_loss, val_acc, PREDS, LABELS = validate()
        
        # append training and validation loss
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        acc_dict['train'].append(train_acc)
        acc_dict['val'].append(val_acc)
        logger.info(f'  Training acc:{train_acc:.3f}, Training Loss: {train_loss:.3f}')
        logger.info(f'  Validation acc:{val_acc:.3f}, Validation Loss: {val_loss:.3f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            D = {'texts': valid_texts , 'labels': trans(LABELS), 'preds':trans(PREDS)}
            validation_preds = pd.DataFrame.from_dict(D)
            validation_preds.to_csv(os.path.join(args.save_dir, "dev_predict.csv"))
            logger.info(f"  Saving model to {args.save_dir}.")
            model.save_pretrained(args.save_dir)

    end = time.time()
    Min = (end - start)//60
    Sec = (end-start) % 60
    logger.info(f'  Running {Min} min(s) and {(Sec):.2f} seconds.')
    logger.info(f'  The saved model\'s val acc is {best_acc:.3f}.')
    logger.info(f'  Finished.')

if __name__ == "__main__":
    args = parse_args()
    main(args)