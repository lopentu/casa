import more_itertools as mit
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import TensorDataset, DataLoader
import time
import re
import numpy as np
import torch

# from collections import OrderedDict
class MTBert:

      def __init__(self):
          self.model_path = ''
          self.model_name = 'bert-base-chinese'
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
          self.label_list = {0: 'B-VN', 1:'B-VP',  2:'I-VN', 3: 'I-VP', 4:'O'} # property: public, private, protected
          self.spectoks = [101, 102, 103, 0] # Don't take away [UNK] 
                  
      def load(self, path):
          '''loading model and tokenizer'''
          self.model_path = path 
          self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
          self.model = BertForTokenClassification.from_pretrained(self.model_path, num_labels = 5)
          self.model.to(self.device)
          print('Finished Loading.')
      
      @staticmethod
      def findspans(texts, true_predictions):
          '''input: texts: a list of strings, true_predictions: a list of list of labels
            return: (span_idxs, spans)'''
          assert len(texts) == len(true_predictions)
          spans_forall=[]
          spanidx_forall=[]
          spanpols_forall = []
          for text, pred in zip(texts, true_predictions):
                spans_for_one_text = []
                pol_for_one = []
                span_idx = []
                pred = pred[:len(text)]
                pairs = [(i, val) for i, val in enumerate(pred) if val != 'O']
                pols = []
                if len(pairs) >1:
                    
                    idxes = [x for (x, val) in pairs]
                    idxes_groups = [list(group) for group in mit.consecutive_groups(idxes)]
                    pol_groups = [pred[group[0]][-1] for group in idxes_groups]
                    for group, pgroup in zip(idxes_groups, pol_groups):
                        label = ''.join([text[i] for i in group])
                        # clean up trailing punctuations
                        label = re.sub(r"^[，。！!\"#$%&\'()*+,-./:;<=>?@[\]\^_`{|}~]", '',label)
                        label = re.sub(r"[，。！!\"#$%&\'()*+,-./:;<=>?@[\]\^_`{|}~]$", '', label)
                        if len(label)>0: 
                            spans_for_one_text.append(label)   
                            span_idx.append(group)
                            pol_for_one.append(pgroup)
                    
                spans_forall.append(spans_for_one_text)
                spanidx_forall.append(span_idx)
                spanpols_forall.append(pol_for_one)
                del spans_for_one_text, span_idx, pol_for_one
          # adding back to span dictionary 
          return spanidx_forall, spans_forall, spanpols_forall
          
      @staticmethod 
      def summarize(input, rec1, rec2, spidx, spans, sppols, isbatch):
          # rec1 = {'input_ids':[], 'preds':[], 'logits':[]}
          # rec2 = {'preds':[], 'probs':[]}
          pols = []
          for pol in rec2['preds']:   
            if pol == 0: pol = 'Neutral'
            elif pol == 1: pol = 'Positive'
            elif pol == 2: pol = 'Negative'
            pols.append(pol)
          assert len(pols) == len(rec2['preds'])
          # else: print(f'An unknown label {label} occurred.')
          if not isbatch: 
            return {'string': input, 'seq_polarity':pol, 'spans': spans[0], 'span_idxs': spidx[0], 'span_pols': sppols[0], \
                    'token_logits':rec1['logits'], 'pol_probs':rec2['probs'] }
          ListofDicts = []
          # print(spans, spidx, sppols)
          for i in range(len(input)):
              Dict = {'string': input[i], 'seq_polarity':pols[i], 'pol_probs':rec2['probs'][i], 'token_logits': rec1['logits'][i],\
                      'spans': spans[i], 'span_idxs':  spidx[i], 'span_pols': sppols[i]}
              ListofDicts.append(Dict)
          del input, rec1, rec2, spidx, spans
          return ListofDicts 
          
      def predict(self, input):
          '''input: string, return: dictionary'''
          self.model.eval()
          input = input.strip()
          X = list(input) # turn the input into a list of elements

          t1records = {'spans':[], 'span_idxs':[]}
          t2records = {'pol':[], 'probs':[]}
          X = self.tokenizer(X, truncation = True, is_split_into_words = True, padding = 'max_length', return_tensors="pt") 
          X.to(self.device)
          with torch.no_grad():
              logits = self.model(**X).logits.view(-1, 5)
          probs = torch.softmax(logits, axis=1).squeeze()
          _, pred = torch.max(probs, -1)
          npprobs = probs.cpu().numpy()
          pred = pred.cpu().numpy()

          # # --- task 1: token extraction ---
          
          # print('Probability shape:', probs.shape)
          # print('prediction shape:', pred.shape)
          t1records = {'pred':[], 'probs':[], 'input_ids':[]} 
          t1records['pred'].extend(pred)
          t1records['input_ids'].extend(x['input_ids'].cpu().numpy().flatten())
          t1records['logits'] = logits.cpu().numpy()

          true_predictions = [[self.label_list[p] for (p, inpid) in zip(t1records['pred'], t1records['input_ids']) if inpid not in self.spectoks]]

          spidx, spans, spanpols = self.findspans([input], true_predictions) 

          # # --- task 2: polarity classification ---
          # getting task 2 prediction
          t2records = {'preds':[]}
          P = probs[:,[1,3]].sum(1)/2
          N = probs[:,[0,2]].sum(1)/2
          O = probs[:,[4]].sum(1)
          merged_probs = torch.stack([O, P, N], dim = 1)[0]

          _, pred = torch.max(merged_probs, -1)
          t2records['preds'].append(pred.cpu().item())
          t2records['probs'] = merged_probs.cpu().numpy()

          # # --- organize ---
          return self.summarize(input, t1records, t2records, spidx, spans, spanpols, isbatch = False) 
      
      def bpredict(self, input, batch_size = 100): 
          '''batch prediction
          input: a list of strings
          return: a list of dictionaries'''
          start = time.time()

          self.model.eval()
          texts = [list(x) for x in input]
          tok_X = tokenizer(
              texts, 
              max_length=512,
              padding ='max_length',
              truncation=True,
              is_split_into_words=True, 
          )

          test_set = TensorDataset(torch.tensor(tok_X['input_ids']), torch.tensor(tok_X['attention_mask']))
          test_loader = DataLoader(test_set, batch_size = batch_size, shuffle=False, pin_memory=True)
          
          t1records = {'input_ids':[], 'preds':[], 'logits':[]}
          t2records = {'preds':[], 'probs':[]}
          for batch in test_loader:
              with torch.no_grad():
                  input_ids = batch[0].to(self.device) # input_ids
                  attention_mask = batch[1].to(self.device)
                  
                  logits = model(input_ids, attention_mask=attention_mask).logits
                  active_loss = attention_mask.view(-1) == 1
                  # getting active logits
                  active_logits = logits.view(-1, 5) 
                  
                  # # ---task1: token extraction --- 
                  
                  _, pred = torch.max(logits , -1)
                  # print('pred shape:', pred.shape)
                  t1records['preds'].extend(pred.cpu().numpy()) # 
                  t1records['input_ids'].extend(input_ids.cpu().numpy()) # for span matching 
                  t1records['logits'].extend(logits.cpu().numpy())
                  
                  # # --- task 2: polarity classification ---
                  
    
                  P = active_logits[:,[1,3]].sum(1)/2 
                  N = active_logits[:,[0,2]].sum(1)/2
                  O = active_logits[:,[4]].sum(1)
                  
                  merged_logits= torch.stack([O, P, N], dim = 1)[::512] #0: O, 1: P, 2: N
                  merged_probs = torch.softmax(merged_logits, axis=1).squeeze()
                  # print('merged_logits shape:', merged_logits.shape)
                  _, pred = torch.max(merged_logits, -1)
                  t2records['preds'].append(pred.cpu().numpy())
                  t2records['probs'].append(merged_probs.cpu().numpy())
          
          
          true_predictions = [[self.label_list[p] for (p, inpid) in zip(predsent, inpidsent) if inpid not in self.spectoks] 
                               for predsent, inpidsent in zip(t1records['preds'], t1records['input_ids'])]
          spidx, spans, spanpols = self.findspans(input, true_predictions) 
          
          t1records['preds']= np.vstack(t1records['preds']) # 
          t1records['input_ids']= np.vstack(t1records['input_ids']) # for span matching 
          
          t2records['preds'] = np.vstack(t2records['preds']).flatten()
          t2records['probs'] = np.vstack(t2records['probs'])
          end = time.time()
          total = end-start
          
          print(f'Total runtime: {total//60} mins {total%60:.3f} secs.')
          return self.summarize(input = input, rec1 = t1records, rec2 = t2records, \
                                spidx = spidx, spans = spans, sppols = spanpols, isbatch = True)
