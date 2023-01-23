import pandas as pd
import requests
from lxml import etree
from bs4 import BeautifulSoup
import re

df = pd.read_excel("/Innolux/US_Publication_List_for_testing.xlsx",engine='openpyxl')
select_df = pd.DataFrame(df)
url_example='https://patents.google.com/patent/'
Abstract = []
Classfication=[]
Description=[]
Claim =[]
title=[]
for i in range(len(select_df)):
  pn = select_df.iloc[i,0]   #抓專利號碼
  url = url_example+pn
  response = requests.get(url)
  soup = BeautifulSoup(response.text,'lxml') 

  try:
    abs = soup.find_all(class_="abstract")
    abs_text= abs[0].text
    Abstract.append(abs_text)

    tit = soup.find('span',itemprop="title")
    tmp=[]
    temp=tit.text
    temp = temp.strip()
    title.append(temp)

    dec = soup.find_all( class_='description-line')
    tmp=[]
    for ab in dec :
      temp=ab.text.lower()
      temp = temp.strip()
      temp = temp.replace('\n', '')
      tmp.append(temp)
    tmp=pd.unique(tmp).tolist()
    tmp = "".join(tmp)
    Description.append(tmp)

    cls = soup.find_all('span',itemprop="Description")
    tmp=[]
    for ab in  cls :
      tmp.append(ab.text.lower())
    tmp=pd.unique(tmp).tolist()
    tmp = ",".join(tmp)
    Classfication.append(tmp)

    
    cla = soup.find_all(class_='claim-text')
    tmp=[]
    for ab in cla :
      tmp.append(ab.text.lower())
    tmp=pd.unique(tmp).tolist()
    tmp = "".join(tmp)
    tmp=tmp.replace('\n', '')
    tmp = re.sub(r'[0-9]+.', '', tmp)
    Claim.append(tmp)
    
  except:                   
    print(pn)

f = open('test_Abstract.txt', 'w') 
for word in Abstract:
  f.write(word+'\n')
f.close()

f = open('test_Classfication.txt', 'w') 
for word in Classfication:
  f.write(word+'\n')
f.close()

f = open('test_Description.txt', 'w') 
for word in Description:
  f.write(word+'\n')
f.close()

f = open('test_Claim.txt', 'w')
for word in Claim:
  f.write(word+'\n')
f.close()

f = open('test_Title.txt', 'w')
for word in title:
  f.write(word+'\n')
f.close()