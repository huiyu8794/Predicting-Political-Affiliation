##############################################################
#
# utils.py
# This file contains various functions that are applied in
# the training loops.
# They convert batch data into tensors, feed them to the models,
# compute the loss and propagate it.
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
# get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time


def my_collate1(batches):
    # return batches
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def loss_fun(outputs, targets):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, targets)
    # return nn.BCEWithLogitsLoss()(outputs, targets)


def evaluate(target, predicted):
    true_label_mask = [1 if (np.argmax(x)-target[i]) ==
                       0 else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction-true_prediction
    accuracy = true_prediction/nb_prediction
    return{
        "accuracy": accuracy,
        "nb exemple": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
    }


def train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []
    for batch_idx, batch in enumerate(data_loader):

        #         model.half()
        #         ids_batch=[data["ids"] for data in batch]
        #         mask_batch=[data["mask"] for data in batch]
        #         token_type_ids_batch = [data["token_type_ids"] for data in batch]
        #         targets_batch = [data["targets"] for data in batch]
        #         lengt_batch=[data['len'] for data in batch]
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]
        lengt = [data['len'] for data in batch]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.cat(targets)
        lengt = torch.cat(lengt)

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 250 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses


def eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):

        #         model.half()
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]
        lengt = [data['len'] for data in batch]
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.cat(targets)
        lengt = torch.cat(lengt)

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(
            outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses
#     return np.vstack(fin_outputs), np.vstack(fin_targets), losses



def rnn_train_loop_fun1(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []
    fin_targets = []
    fin_outputs = []
    for batch_idx, batch in enumerate(data_loader):
        #         model.half()
        #         ids_batch=[data["ids"] for data in batch]
        #         mask_batch=[data["mask"] for data in batch]
        #         token_type_ids_batch = [data["token_type_ids"] for data in batch]
        #         targets_batch = [data["targets"] for data in batch]
        #         lengt_batch=[data['len'] for data in batch]
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        # print(targets)
        lengt = [data['len'] for data in batch]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt = torch.cat(lengt)
        lengt = [x.item() for x in lengt]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
        loss = loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        acc = evaluate(targets.cpu().detach().numpy(), torch.softmax(outputs, dim=1).cpu().detach().numpy())['accuracy']
        if batch_idx % 10 == 0:
            print(f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, accuracy={acc:.2f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses
    


def rnn_eval_loop_fun1(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"][0] for data in batch]
        lengt = [data['len'] for data in batch]
#         ids_batch=[data["ids"] for data in batch]
#         mask_batch=[data["mask"] for data in batch]
#         token_type_ids_batch = [data["token_type_ids"] for data in batch]
#         targets_batch = [data["targets"] for data in batch]
#         lengt_batch=[data['len'] for data in batch]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        targets = torch.stack(targets)
        lengt = torch.cat(lengt)
        lengt = [x.item() for x in lengt]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
            loss = loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def test_eval_loop_fun(data_loader, model, device):
    model.eval()
    fin_outputs = []
    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        # print(targets)
        lengt = [data['len'] for data in batch]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        token_type_ids = torch.cat(token_type_ids)
        lengt = torch.cat(lengt)
        lengt = [x.item() for x in lengt]

#         for doc in range(len(lengt_batch)):
#             ids=ids_batch[doc]
#             mask=mask_batch[doc]
#             token_type_ids=token_type_ids_batch[doc]
#             targets=targets_batch[doc]
#             lengt=lengt_batch[doc]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
            # print(ids)
            # print(mask)
            # print(token_type_ids)
            # print(lengt)
            # print('*'*20)
        fin_outputs.append(np.argmax(torch.softmax(outputs, dim=1).cpu().detach().numpy()))
    return fin_outputs