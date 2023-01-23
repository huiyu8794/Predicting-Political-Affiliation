##############################################################
#
# RoBERT.py
# This file contains the implementation of the RoBERT model
# An LSTM is applied to a segmented document. The resulting
# embedding is used for document-level classification
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

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time


class RoBERT_Model(nn.Module):
    """ Make an LSTM model over a fine tuned bert model.

    Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance

    """
    def __init__(self, BERTwithClassifier, num_classes=2):
        super(RoBERT_Model, self).__init__()
        self.bert = BERTwithClassifier
        self.lstm = nn.LSTM(768, 1024, num_layers=1, bidirectional=False)
        self.linear = nn.Linear(1024, 64)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 350),
            nn.Linear(350, num_classes)
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, ids, mask, token_type_ids, lengt):
        """ Define how to performed each call

        Parameters
        __________
        ids: array
        mask: array
        token_type_ids: array
        lengt: int
        _______
        """
        poo=self.bert(ids, attention_mask=mask,token_type_ids=token_type_ids)['pooler_output']
      
        chunks_emb = poo.split_with_sizes(lengt)

        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        batch_emb_pad = nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=-99, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input, )  # (h_t, h_c))
        h_t = h_t.view(-1, 1024)
        out=self.classifier(h_t)

        return (out)


class BERTwithClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BERTwithClassifier, self).__init__()
        # BERT: language model
        self.bert = BertModel.from_pretrained("bert-base-chinese") #bert-base-uncased
        self.classifier = nn.Sequential(
            nn.Linear(768, 350),
            nn.Linear(350, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        temp = self.bert(input_ids, attention_mask, token_type_ids)['pooler_output']
        output = self.classifier(temp)

        return output