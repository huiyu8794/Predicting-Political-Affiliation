import nltk

Abstract = []
Classfication=[]
Description=[]
Claim =[]
title=[]
max_len=500
f = open('Innolux/Abstract.txt', 'r', encoding='utf-8')
for line in f.readlines():
                token = nltk.word_tokenize(line)
                if len(token)>max_len:
                    token = token[0:max_len]
                token=' '.join(token)
                Abstract.append(token)
f.close()
f = open('Innolux/Claim.txt', 'r', encoding='utf-8')
for line in f.readlines():
                token = nltk.word_tokenize(line)
                if len(token)>max_len:
                    token = token[0:max_len]
                token=' '.join(token)
                Claim.append(token)
f.close()
f = open('Innolux/Classfication.txt', 'r', encoding='utf-8')
for line in f.readlines():
                token = nltk.word_tokenize(line)
                if len(token)>max_len:
                    token = token[0:max_len]
                token=' '.join(token)
                Classfication.append(token)
f.close()
f = open('Innolux/Description.txt', 'r', encoding='utf-8')
     
for line in f.readlines():
                token = nltk.word_tokenize(line)
                if len(token)>max_len:
                    token = token[0:max_len]
                token=' '.join(token)
                Description.append(token)
f.close()
f = open('Innolux/Title.txt', 'r', encoding='utf-8')
        
for line in f.readlines():

                title.append(line)
f.close()
total=[]
for i in range(len(Abstract)):
    tmp='title '+title[i]+' Abstract '+Abstract[i]+'Description '+Description[i][0:500]+'Claim '+Claim[i][0:500]+Classfication[i][0:500]
    tmp=tmp.replace('\n', '')
    total.append(tmp)



f = open('Innolux/total.txt', 'w')
for word in total:
  f.writelines(word+'\n')
f.close()
