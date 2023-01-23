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
from Custom_Dataset_Class import ConsumerComplaintsDataset1
from Bert_Classification import Bert_Classification_Model
from RoBERT import RoBERT_Model, BERTwithClassifier
import seaborn as sns
from BERT_Hierarchical import BERT_Hierarchical_Model
import warnings
warnings.filterwarnings("ignore")
import nltk
import os
nltk.download('punkt')

if torch.cuda.is_available():    
    device_id = 'cuda:1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device_id = torch.device('cpu')

TRAIN_BATCH_SIZE = 1
EPOCH = 30
validation_split = .1
shuffle_dataset = True
random_seed = 777

path = '/home/huiyu8794/pp_final/data/training_news.csv'

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

dataset=ConsumerComplaintsDataset1(
    tokenizer=bert_tokenizer,
    min_len=0,
    max_len=2000,
    chunk_len=50,
    overlap_len=0,
    file_location=path
    )

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler,
    collate_fn=my_collate1)

valid_data_loader=DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=my_collate1)

num_training_steps = int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

model = BERTwithClassifier(num_classes=3)
# model.load_state_dict(torch.load('/home/Innolux/save.pt'))
model_rnn = RoBERT_Model(list(model.children())[0], num_classes=3).to(device_id)
optimizer = AdamW(model_rnn.parameters(),lr=5e-6)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = num_training_steps)
val_losses=[]
batches_losses=[]
val_acc=[]
train_acc=[]
val_acc_tmp=0
def plt_confusion_matrix(output, target, epoch):
    labels = ['blue', 'green']
    a = confusion_matrix(target.reshape(-1),np.argmax(output,axis=1))
    sns.set()
    f,ax=plot.subplots(figsize=(10,10))
    sns.heatmap(a, annot=True, ax=ax, xticklabels=labels, yticklabels=labels) #画热力图
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plot.savefig('/home/huiyu8794/pp_final/save/output_best.png')

for epoch in range(EPOCH):
        t0 = time.time()    
        print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
        output, target, batches_losses_tmp = rnn_train_loop_fun1(train_data_loader, model_rnn, optimizer, device_id)
        epoch_loss = np.mean(batches_losses_tmp)
        tmp_train = evaluate(target.reshape(-1), output)
        train_acc.append(tmp_train['accuracy'])
        print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
        t1=time.time()
        output, target, val_losses_tmp = rnn_eval_loop_fun1(valid_data_loader, model_rnn, device_id)
        print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")    
        tmp_evaluate=evaluate(target.reshape(-1), output)
        print(f"=====>\t{tmp_evaluate}")
        val_acc.append(tmp_evaluate['accuracy'])
        val_losses.append(val_losses_tmp)
        batches_losses.append(batches_losses_tmp)
        if tmp_evaluate['accuracy'] >= val_acc_tmp:
            val_acc_tmp=tmp_evaluate['accuracy']
            print("\t§§ the RNN model has been saved §§")
            plt_confusion_matrix(output, target, epoch)
            torch.save(model_rnn.state_dict(), '/home/huiyu8794/pp_final/save/best_model.pt') 
print('final val_acc_tmp=', val_acc_tmp)   

output, target, val_losses_tmp = rnn_eval_loop_fun1(valid_data_loader, model_rnn, device_id)
pd.DataFrame(np.array([[np.mean(x) for x in batches_losses], [np.mean(x) for x in val_losses]]).T,
                        columns=['Training', 'Validation']).plot(title="loss")
plot.show()
plot.savefig('/home/huiyu8794/pp_final/save/loss.png')
pd.DataFrame(np.array([train_acc,val_acc]).T, 
                       columns=['Training', 'Validation']).plot(title="accuracy")
plot.show()
plot.savefig('/home/huiyu8794/pp_final/save/accuracy.png')

  

