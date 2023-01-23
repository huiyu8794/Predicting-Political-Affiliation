##############################################################
#
# Custom_Dataset_Class.py
# This file contains the code to load and prepare the dataset
# for use by BERT.
# It does preprocessing, segmentation and BERT features extraction
#
##############################################################

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
import warnings
warnings.filterwarnings("ignore")
import nltk
# nltk.download('punkt')
class ConsumerComplaintsDataset1(Dataset):
    """ Make preprocecing, tokenization and transform consumer complaints
    dataset into pytorch DataLoader instance.

    Parameters
    ----------
    tokenizer: BertTokenizer
        transform data into feature that bert understand
    max_len: int
        the max number of token in a sequence in bert tokenization. 
    overlap_len: int
        the maximum number of overlap token.
    chunk_len: int
        define the maximum number of word in a single chunk when spliting sample into a chumk
    approach: str
        define how to handle overlap token after bert tokenization.
    max_size_dataset: int
        define the maximum number of sample to used from data.
    file_location: str
        the path of the dataset.

    Attributes
    ----------
    data: array of shape (n_keept_sample,)
        prepocess data.
    label: array of shape (n_keept_sample,)
        data labels
    """

    def __init__(self, tokenizer, max_len, overlap_len, file_location,chunk_len, min_len,approach="all"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.file_location=file_location
        self.data, self.label = self.process_data(file_location,tokenizer,max_len)

    def process_data(self, file_location, tokenizer, max_len):


        # f = open(file_location, 'r', encoding='utf-8')
        # token_list = []
        # for line in f.readlines():
        #         token = nltk.word_tokenize(line)
        #         if len(token) > self.max_len:
        #             token = token[0:self.max_len]
        #             print('too many')
        #         token_list.append(' '.join(token))
        # f.close()

        with open(file_location, newline='') as csvfile:
            # rows = csv.reader(csvfile)
            df = pd.read_csv(csvfile)
            rows = df['comment']
            token_list = []
            for row in rows:
                token = nltk.word_tokenize(row)
                # if len(token) > self.max_len:
                #     token = token[0: self.max_len]
                #     print('too many')
                token_list.append(' '.join(token))

        token_list = np.array(token_list)
       
        #fake label
        label_list = []
        for i in range(len(token_list)):
            label_list.append(i)

        return token_list, label_list


    def clean_txt(self, text):
        """ Remove special characters from text """

        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def long_terms_tokenizer(self, data_tokenize, targets):
        """  tranfrom tokenized data into a long token that take care of
        overflow token according to the specified approch.

        Parameters
        ----------
        data_tokenize: dict
            an tokenized result of a sample from bert tokenizer encode_plus method.
        targets: array
            labels of each samples.

        Returns
        _______
        long_token: dict
            a dictionnary that contains
             - [ids]  tokens ids
             - [mask] attention mask of each token
             - [token_types_ids] the type ids of each token. note that each token in the same sequence has the same type ids
             - [targets_list] list of all sample label after add overlap token as sample according to the approach used
             - [len] length of targets_list
        """

        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        remain =  data_tokenize.get("overflowing_tokens").reshape(-1)
        targets = torch.tensor(targets, dtype=torch.int)

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)

        if self.approach != 'head':
        # if remain:
            remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain)+self.chunk_len)
            idxs = idxs[(self.chunk_len-self.overlap_len-2)
                         ::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap = previous_input_ids[-(
                self.overlap_len+1):-1]
            start_token = torch.tensor([101], dtype=torch.long)
            end_token = torch.tensor([102], dtype=torch.long)
            
            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = torch.cat(
                        (input_ids_first_overlap, remain[:idx]))
                  
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx-self.overlap_len):idx]
        

                previous_idx = idx

                nb_token = len(input_ids)+2
                attention_mask = torch.ones(self.chunk_len, dtype=torch.long)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = torch.zeros(self.chunk_len, dtype=torch.long)
                input_ids = torch.cat((start_token, input_ids, end_token))
                
                if self.chunk_len-nb_token > 0:
                    padding = torch.zeros(
                        self.chunk_len-nb_token, dtype=torch.long)
                    input_ids = torch.cat((input_ids, padding))
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)
            if self.approach == "tail":
                input_ids_list = [input_ids_list[-1]]
                attention_mask_list = [attention_mask_list[-1]]
                token_type_ids_list = [token_type_ids_list[-1]]
                targets_list = [targets_list[-1]]
        return({
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            'mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })


    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""
        text=['sposed on the first substrate at a same distance from the first substrate as the first sensor electrode. The']
        
        consumer_complaint = str(self.data[idx])
        targets = int(self.label[idx])
        data = self.tokenizer.encode_plus(
            consumer_complaint,
            max_length=self.chunk_len,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt',truncation = True)
       
        long_token = self.long_terms_tokenizer(data, targets)
        return long_token

    def __len__(self):
        """ Return data length """
        return len(self.data)
