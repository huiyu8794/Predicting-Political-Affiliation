import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plot
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
from sklearn.metrics import confusion_matrix
from utils import *
from Custom_Dataset_Class_for_testing import ConsumerComplaintsDataset1
from Bert_Classification import Bert_Classification_Model
from RoBERT import RoBERT_Model,BERTwithClassifier
import seaborn as sns
from BERT_Hierarchical import BERT_Hierarchical_Model
import os
import warnings
from multiprocessing import Pool
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
os.environ["LOCAL_RANK"]="0,1"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device_id = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# if torch.cuda.is_available():    
#     device_id = 'cuda:1'
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
#     device_id = torch.device("cpu")

model = BERTwithClassifier()
# model.load_state_dict(torch.load('/home/Innolux/save.pt'))
model_rnn = RoBERT_Model(list(model.children())[0]).to(device_id)
# model_rnn.load_state_dict(torch.load('/home/run_best.pt'))
# model_rnn = nn.DataParallel(model_rnn)

TRAIN_BATCH_SIZE=1
EPOCH=10
validation_split = .1
shuffle_dataset = True


load_path='/home/huiyu8794/pp_final/data/crawler_result.csv'

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

dataset = ConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=0,
    max_len=2000,
    chunk_len=50,
    overlap_len=0,
    file_location=load_path
    )

# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# test_data_loader = DataLoader(
#     dataset,
#     batch_size=TRAIN_BATCH_SIZE,
#     collate_fn=my_collate1)

val_losses=[]
batches_losses=[]
val_acc=[]
train_acc=[]
val_acc_tmp=0

# 只 master 进程做 logging，否则输出会很乱
if args.local_rank == 0:
    tb_writer = SummaryWriter(comment='ddp-training')

# 分布式数据集
train_sampler = DistributedSampler(dataset)
train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE, collate_fn=my_collate1)

# 分布式模型
model_rnn = torch.nn.parallel.DistributedDataParallel(model_rnn, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

############
start_time = time.time()
output = test_eval_loop_fun(train_loader, model_rnn, device_id)
end_time = time.time()
    
print(f"{end_time - start_time} 秒預測 {len(output)} 筆的資料")

predict = []

g = 0
b = 0
for i in range(len(output)):

    if output[i]==0:
      predict.append("blue")
      b = b+1

    elif output[i]==1:
      predict.append("green")
      g = g+1

    else:
      predict.append("neutral")
    
# df = pd.read_csv(load_path)
# pat = df['comment']    
# dict = {'title': pat, "predict label": predict} 
# df = pd.DataFrame(dict) 
# df.to_excel('/home/huiyu8794/pp_final/save/test_output.xlsx')

if args.local_rank == 0:
  torch.distributed.barrier()

if b>g:
  print('Blue')
else:
  print('Green')


