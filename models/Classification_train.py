import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import sys
import os
import numpy as np
import pandas as pd 
from transformers import AutoModel, BertTokenizerFast, AutoModelForSequenceClassification, BertConfig, DataCollatorWithPadding
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import random
from transformers import AdamW, set_seed
import time
import ipdb
from utils import normalize_text
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import wandb


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Classification task")
    parser.add_argument(
        "--train_file", type=str, default='./data/Train_risk_classification_ans.csv', help="A file containing the training data."
    )
    parser.add_argument(
        "--eval_file", type=str, default='./data/Develop_risk_classification.csv', help="A file containing the eval data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="Preproc num workers"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='ckiplab/albert-base-chinese',
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='bert-base-chinese',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--save_model_dir", type=str, default='./task1_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1027, help="A seed for reproducible training.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=200,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="The ID of the device you would like to use.",
    )
    parser.add_argument(
        "--train_full",
        action="store_true",
        help="Training with full training data.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability to apply.",
    )
    args = parser.parse_args()

    if args.save_model_dir is not None:
        os.makedirs(args.save_model_dir, exist_ok=True)
    return args


def main(args):
    wandb.init(project='ntu_nlp_risk', entity='pwlin')
    config = wandb.config
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.dropout = args.dropout
    config.model = args.model_name
    config.batch_size = args.per_device_train_batch_size
    config.device = args.device


    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples['text'],
            truncation=True, 
            max_length=args.max_seq_length,  # max_seq_length
            stride=args.doc_stride,   
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length'  # "max_length"
        )   
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples.pop("offset_mapping")
        # Let's label those examples!
        tokenized_examples["labels"] = []  # 0 or 1 
        tokenized_examples["example_id"] = []
        for sample_index in sample_mapping:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["article_id"][sample_index])
            tokenized_examples['labels'].append(int(examples['label'][sample_index]))
        return tokenized_examples

    raw_dataset = datasets.load_dataset("csv", data_files=args.train_file)
    train_dataset = raw_dataset['train'].map(normalize_text)
    raw_dataset = raw_dataset['train'].train_test_split(test_size=0.03)
    # if args.debug:
    #     for split in raw_dataset.keys():
    #         raw_dataset[split] = raw_dataset[split].select(range(20))

    if not args.train_full:
        train_dataset = raw_dataset['train'].map(normalize_text)
    eval_dataset = raw_dataset['test'].map(normalize_text)
    column_names = raw_dataset["train"].column_names
    train_ids = train_dataset['article_id']
    eval_ids = eval_dataset['article_id']
    num_train_samples = len(train_dataset)
    num_eval_samples = len(eval_dataset)
    num_samples = num_train_samples + num_eval_samples
    train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
    )
    eval_dataset = eval_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    num_train_batch = len(train_dataloader)
    num_eval_batch = len(eval_dataloader)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(args.device)
    model.config.attention_probs_dropout_prob = args.dropout
    model.config.hidden_dropout_prob = args.dropout
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    torch_softmax = nn.Softmax(dim=1)
    best_auroc = float('-inf')
    wandb.watch(model)
    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        y_preds = np.zeros((num_samples+1, 2))
        y_trues = np.zeros(num_samples+1)
        for step, batch in enumerate(train_dataloader):
            example_ids = batch.pop('example_id').tolist()
            # print(example_ids)
            # exit()
            for i in batch.keys():
                batch[i] = batch[i].to(args.device)
            outputs = model(**batch)
            
            y_pred = torch_softmax(outputs.logits).cpu().data.numpy()
            y = batch.labels.cpu().data.numpy()
            for i, example_id in enumerate(example_ids):
                y_preds[example_id][0] += np.log(y_pred[i][0])
                y_preds[example_id][1] += np.log(y_pred[i][1])
                y_trues[example_id] = y[i]
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            if (step+1) % args.gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            print(f'[{step:3d}/{num_train_batch}]',end='\r')
        train_acc = (np.sum(np.argmax(y_preds, axis=1) == y_trues) - num_eval_samples - 1)/num_train_samples
        train_loss /= num_train_batch
        

        model.eval()
        eval_loss = 0
        y_preds = np.zeros((num_samples+1,2))
        y_trues = np.zeros(num_samples+1)
        y_preds = []
        y_trues = []
        last_example_id = -1
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                example_ids = batch.pop('example_id')
                for i in batch.keys():
                    batch[i] = batch[i].to(args.device)
                outputs = model(**batch)
                y_pred = torch_softmax(outputs.logits).cpu().data.numpy()
                y = batch.labels.cpu().data.numpy()
                for i, example_id in enumerate(example_ids):
                    # y_preds[example_id][0] += np.log(y_pred[i][0])
                    # y_preds[example_id][1] += np.log(y_pred[i][1])
                    if example_id != last_example_id:
                        y_trues.append(y[i])
                        y_preds.append([0, 0])
                    # zero_score = 1 if y_pred[i][0] > y_pred[i][1] else 0
                    # one_score = 1 - zero_score
                    y_preds[-1][0] += np.log(y_pred[i][0])
                    y_preds[-1][1] += np.log(y_pred[i][1])
                    last_example_id = example_id

                loss = outputs.loss
                eval_loss += loss.item()
        # sum logP
        # eval_acc = (np.sum(np.argmax(y_preds, axis=1) == y_trues) - num_train_samples - 1)/num_eval_samples
        try:
            assert len(y_trues) == len(y_preds)
        except:
            ipdb.set_trace()
        eval_acc = np.sum(np.argmax(np.array(y_preds), axis=1) == np.array(y_trues)) / len(y_trues)
        eval_auroc = roc_auc_score(np.array(y_trues), softmax(np.array(y_preds), axis=1)[:, 1])
        eval_loss /= num_eval_batch

        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs:02d}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        print(f' eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f}, eval auroc: {eval_auroc:.4f}')
        wandb.log({
            "train loss": train_loss,
            "train acc": train_acc,
            "eval loss": eval_loss,
            "eval acc": eval_acc,
            "eval auroc": eval_auroc
        })
        
        if eval_auroc > best_auroc:
            best_auroc = eval_auroc
            model.save_pretrained(args.save_model_dir)
            print(f"Saving model at eval auroc {eval_auroc:.4f}")

    print('Done')
    return

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        set_seeds(args.seed)
    main(args)

