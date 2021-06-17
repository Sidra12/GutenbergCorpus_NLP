#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import matplotlib.pyplot as plt
import io
import nltk
#nltk.download('stopwords')
import re
plt.show()
from sklearn.feature_extraction.text import *
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import *
from numpy import argmax
import warnings
warnings.filterwarnings("ignore")
from spacy.displacy.render import EntityRenderer
from nltk.tree import Tree
#from IPython.core.display import display, HTML

import os
import nltk
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk import word_tokenize
import codecs
import pandas as pd
from pandas import Series

dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/MLProject/lemmatized_content.csv')
dataset.head()


# In[2]:


dataset['cleaned_Data_Content'] = dataset['cleaned_Data_Content'].replace(np.nan, 0)


# In[3]:


print(dataset.iloc[[465]]) 


# In[4]:


pip install BeautifulSoup4


# In[5]:



text = []
p_list = []

for i, row in dataset.iterrows():
    sample = dataset['Data_Content'].iloc[i]
    text.append(sample)
    soup = BeautifulSoup(sample)
    p_tag = len(soup.find_all('p'))
    p_list.append(p_tag)

from pandas import Series

dataset_ptag = pd.DataFrame({'p_tag': p_list})
dataset_ptag


# In[6]:


dataset['p_tag_count'] = dataset_ptag['p_tag']


# In[7]:


dataset.head()


# In[8]:


dataset.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/ptagCount_gutenberg.csv', header=None, index=None, sep=';', mode='a')


# In[9]:


dataset['comma_count'] = dataset.Data_Content.str.count(',')
dataset.head()


# In[10]:


dataset['colon_count'] = dataset.Data_Content.str.count(':')
dataset.head()


# In[40]:


dataset['semicolon_count'] = dataset.Data_Content.str.count(';')
dataset.head()


# In[11]:


dataset['hyphen_count'] = dataset.Data_Content.str.count('-')
dataset.head()


# In[13]:


dataset1 = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/TTR_gutenberg.csv')
dataset1.head()


# In[14]:


dataset['TTR'] = dataset1['TTR']
dataset.head()


# In[33]:


dataset2 = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/char_map.csv', sep=';', header= None)
dataset2.head()


# In[35]:


dataset['named_entities'] = dataset2[1]
dataset.head()


# In[26]:


dataset3 = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/sentiments_gutenberg.csv', sep = ';')
dataset3.head()


# In[43]:


dataset['Compound'] = dataset3['Compound']
dataset.head()

dataset['Positive'] = dataset3['Positive']
dataset.head()

dataset['Neutral'] = dataset3['Neutral']
dataset.head()

dataset['Negative'] = dataset3['Negative']
dataset.head()

dataset['BookName'] = dataset3['BookName']
dataset.head()

dataset.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/Gutenberg_Features.csv', index=None, sep=';', mode='a')


# In[44]:


dataset4 = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/FleschKscore.csv')
dataset4.head()


# In[45]:


dataset['FleschK_score'] = dataset4['FleschK']
dataset.head()


# In[46]:


dataset.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/Gutenberg_Features.csv', index=None, sep=';', mode='a')


# In[50]:


dataset_pos = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/pos_tag.csv', sep=';')
dataset_pos.head()


# In[51]:


dataset['CC'] = dataset_pos['CC']
dataset['PRP'] = dataset_pos['PRP']
dataset['PRP$'] = dataset_pos['PRP$']
dataset['IN'] = dataset_pos['IN']
dataset['NNP'] = dataset_pos['NNP']

dataset.head()


# In[53]:


dataset_features = pd.DataFrame

dataset['period_count'] = dataset.Data_Content.str.count('.')
dataset.head()


# In[66]:


dataset_features = dataset[['Target_Class','p_tag_count','comma_count','colon_count','hyphen_count','TTR','Compound','Positive','Neutral','Negative','FleschK_score','CC','PRP','PRP$','IN','NNP','period_count']]


# In[67]:


dataset_features.head()


# In[56]:


dataset_features.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/Gutenberg_Features.csv', index=None, sep=';', mode='a')


# In[68]:


dataset_features['BookName']= dataset['BookName']

dataset_features.head()


# In[103]:


score = dataset_features.groupby(['TTR','BookName'],as_index=False)
score.mean().sort_values(by='TTR', ascending = False)

dataset_df1 = score.mean().sort_values(by='TTR', ascending = False)

dataset_df1.head(6)


# In[108]:


import seaborn as sns
sns.set()
dataset_df1 = score.mean().sort_values(by='TTR', ascending = False)
dataset_features = dataset_df1[0:5]
dataset_df1.head(5)

plt.figure(figsize=(100, 50))
sns.palplot(sns.color_palette("husl", 8))

plt.title('Top 5 Type-Token Ratio Score:')

plt.show()

dataset_features.head(6)
sns.barplot(data = dataset_features
            ,x = 'TTR'
            ,y = 'BookName'
            ,color = 'lightgreen' 
            ,ci = None
            )


# In[ ]:




