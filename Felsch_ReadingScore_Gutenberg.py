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


# In[2]:


pip install empath


# In[3]:


from empath import Empath
lexicon = Empath()


# In[6]:


#lexicon.analyze("he hit the other person", normalize=True)


# In[3]:


dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/MLProject/lemmatized_content.csv')
dataset.head()


# In[8]:


import sys, csv
csv.field_size_limit(sys.maxsize)

# A counter for female and male connections
from collections import Counter
cnt = {'he': Counter(), 'she': Counter()}


# In[16]:


from nltk.util import ngrams
dataset['bigrams'] = dataset['Tokenized_Data_Content'].apply(lambda row: list(nltk.ngrams(row, 2)))


# In[19]:


bigrams = dataset['bigrams']

for gram in bigrams:

        t0 = str(gram[0]) # First token in bigram
        t1 = str(gram[1]) # Second token in bigram

        if t0 == 'han':
            cnt['he'][t1] += 1
        elif t1 == 'han':
            cnt['he'][t0] += 1

        elif str(t0) == 'male_name':
            cnt['he'][t1] += 1
        elif str(t1) == 'male_name':
            cnt['he'][t0] += 1

        elif str(t0) == 'hon':
            cnt['she'][t1] += 1
        elif str(t1) == 'hon':
            cnt['she'][t0] += 1

        elif str(t0) == 'female_name':
            cnt['she'][t1] += 1
        elif str(t1) == 'female_name':
            cnt['she'][t0] += 1


# In[ ]:


he = (
    pd.DataFrame()
    .from_dict(cnt['he'], orient='index')
    .rename(columns={0: 'he'})
    .pipe(lambda d.: d/d.sum())
)

she = (
    pd.DataFrame()
    .from_dict(cnt['she'], orient='index')
    .rename(columns={0: 'she'})
    .pipe(lambda d: d/d.sum())
)


# In[57]:


pip install https://github.com/andreasvc/readability/tarball/master


# In[56]:


text = dataset['cleaned_Data_Content'][1]


# In[4]:


pip install spacy-readability


# In[5]:


dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/lemmatized_content.csv')

dataset.head()



# In[225]:


dataset1.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/set.csv', header=None, index=None, sep=';', mode='a')


# In[8]:


dataset.iloc[[622]] = dataset.iloc[[622]].replace(np.nan, '0')
dataset.iloc[[465]] = dataset.iloc[[465]].replace(np.nan, '0')

dataset.iloc[[465]]


# In[9]:


import spacy
from spacy_readability import Readability

sp = spacy.load('en')
sp.max_length = 9000000
sp.add_pipe(Readability())



read_list = []
text = []
#sample = sp(dataset['cleaned_Data_Content'].iloc[i])


for i, row in dataset.iterrows():
    sample = sp(dataset['cleaned_Data_Content'].loc[i])
    text.append(sample)
    read_list.append(sample._.flesch_kincaid_reading_ease)
    print(sample._.flesch_kincaid_reading_ease)
    
from pandas import Series

dataset_readable = pd.DataFrame({'FleschK': read_list})

#print(dataset_readble.head())


# In[239]:





# In[10]:


dataset_readable.head()


# In[245]:


dataset['FleschKscore'] = dataset_readable2['FleschK']


# In[11]:


dataset_readable.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/FleschKscore.csv', index=None, sep=';', mode='a')


# In[248]:


dataset.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/GutenbergFleschScore.csv', header=None, index=None, sep=';', mode='a')


# In[ ]:




