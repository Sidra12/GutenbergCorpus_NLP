#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[1]:


pip install lexicalrichness


# In[4]:


dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/lemmatized_content.csv')
#dataset2 = dataset[0:40]

from lexicalrichness import LexicalRichness as lexicalrichness

lexical_list = []
text = []
ttr = []
sample = dataset['cleaned_Data_Content']
for i, row in dataset.iterrows():
    sample = dataset['cleaned_Data_Content'].iloc[i]
    lex = lexicalrichness(str(sample))
    text.append(lex)
    ttr = lex.ttr
    lexical_list.append(ttr)
    
from pandas import Series

dataset_readable = pd.DataFrame({'TTR': lexical_list})


# In[5]:


dataset_readable.head()


# In[27]:


text = dataset['cleaned_Data_Content'][25]

lex = lexicalrichness(text)
print(lex.ttr)


# In[6]:


dataset_readable.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/TTR_gutenberg.csv', index=None, sep=',' )


# In[8]:


dataset_readable.head()


# In[13]:


datasetfeat = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/Gutenberg_Features.csv', sep=';')
datasetfeat.head()


# In[15]:



score = datasetfeat.groupby(['TTR','BookName'],as_index=False)
score.mean().sort_values(by='TTR', ascending = False)



# In[ ]:


import seaborn as sns
sns.set()
dataset_df1 = score.mean().sort_values(by='TTR', ascending = False)
dataset_feature = dataset_df1[0:6]
dataset_df1.head(6)

plt.figure(figsize=(100, 50))
sns.palplot(sns.color_palette("husl", 8))

plt.title('Top 5 Flesch Kincaid Reading Ease Score:')

plt.show()

dataset_df1.head(6)
sns.barplot(data = dataset_feature
            ,x = 'FleschK_score'
            ,y = 'BookName'
            ,color = 'skyblue' 
            ,ci = None
            )


# In[ ]:




