#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

import os
import nltk
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk import word_tokenize
import codecs
import pandas as pd

cwd = os.getcwd()
#testing with codecs to open a html file, not necessarily needed though.
import codecs
data1 = codecs.open('/Users/sidraaziz/PycharmProjects/MLProject/Gutenberg_19th_century_English_Fiction/pg11CarolAlice-content.html','r')
#print(data1.read())

dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/master996.csv', sep=';', engine='python')

row_count = len(dataset.axes[0])
data_list = []  # list containing all text data
data_class = []  # class of the respective text data retrieved

for i in range(0, row_count):
    bookid = dataset.iloc[i, 1]
    bookid_split = bookid.split('.')
    data_class.append(dataset.iloc[i, 2])
    url = 'file:///' + '/Users/sidraaziz/PycharmProjects/ProjectGutenberg/Gutenberg_19th_century_English_Fiction/' + bookid_split[0] + '-content.html'
    data_list.append(urllib.request.urlopen(url).read())[]
   # print(data_list[0])

#adding the data_list and data_class to dataframe
dictionary = {'Data_Content':data_list,'Target_Class': data_class}
dataset = pd.DataFrame(dictionary, columns=['Data_Content','Target_Class'])
#print(dataset.head())

#Now dataframe created above can be split into Features (X) and Target class (y)
le_target = LabelEncoder()
obj_dataset= dataset.select_dtypes(include=['object']).copy()
obj_dataset['genreLabel'] = le_target.fit_transform(obj_dataset['Target_Class'])
print()
#print('below is the list seperation with Target_Class encoded:')
#print(obj_dataset[['genreLabel', 'Target_Class']])
features = ['Data_Content']
# Separating out the features
X = obj_dataset.loc[:, features].values
# Separating out the target
y = obj_dataset.loc[:,['Target_Class']].values

#Next Steps: Work on feature extraction, train and test split of the data.

#onehotencoding on above label encoder
onehot_encoder = OneHotEncoder(sparse=False)
obj_dataset['genreLabel'] = obj_dataset['genreLabel'].values.reshape(len(obj_dataset['genreLabel']), 1)
onehot_encoded = onehot_encoder.fit_transform(obj_dataset['genreLabel'].values.reshape(-1,1))
#print(onehot_encoded)

# invert first example
inverted = le_target.inverse_transform([argmax(onehot_encoded)])
#print (onehot_encoded[970])

#removing special characters, extra whitespaces, digits, stopwords and lower casing the text corpus
wordpunc_tokenize = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#converting data into lower case:
#obj_dataset['Data_Content'] = obj_dataset['Data_Content'].str.lower()
example_lower = obj_dataset.iloc[0]
#print("this is lowercase implementation example:")
print(example_lower)


# In[3]:


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import nltk


# In[2]:


print(obj_dataset.head())


# In[4]:


obj_dataset = pd.DataFrame(obj_dataset) 


# In[ ]:





# In[6]:


obj_dataset.describe() 


# In[77]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

all_content = obj_dataset['Data_Content']
all_sent_values = []
all_sentiments = []

def sentiment_value(paragraph):
    paragraph = str(paragraph)
    analyser = SentimentIntensityAnalyzer()
    result = analyser.polarity_scores(paragraph)
    score = result['compound']
    return (score,1)


sample = obj_dataset['Data_Content'][9]
#print(sample)
print('Sentiment: ')
print(sentiment_value(str(sample)))


# In[58]:


analyzer = SentimentIntensityAnalyzer()

sample = obj_dataset['Data_Content'][2]
compound = analyzer.polarity_scores(str(sample))['compound']
pos = analyzer.polarity_scores(str(sample))['pos']
neu = analyzer.polarity_scores(str(sample))['neu']
neg = analyzer.polarity_scores(str(sample))['neg']

print(' — — — — — — — — -')
print(f'compound: {compound}')
print(f'pos: {pos}')
print(f'neu: {neu}')
print(f'neg: {neg}')


# In[71]:


obj_dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/lemmatized_content.csv')
obj_dataset = pd.DataFrame(obj_dataset)
print(obj_dataset.head())


# In[122]:


sample = obj_dataset['cleaned_Data_Content']
text = []
#number_favourites = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []
#obj_dataset = obj_dataset.str()

analyzer = SentimentIntensityAnalyzer()

for i, row in obj_dataset.iterrows():
    sample = obj_dataset['cleaned_Data_Content'].iloc[i]
    text.append(sample)
    vs_compound.append(analyzer.polarity_scores(str(sample))['compound'])
    vs_pos.append(analyzer.polarity_scores(str(sample))['pos'])
    vs_neu.append(analyzer.polarity_scores(str(sample))['neu'])
    vs_neg.append(analyzer.polarity_scores(str(sample))['neg'])
    
print(obj_dataset.head())


# In[126]:


from pandas import Series

dataset_df = pd.DataFrame({'cleaned_Data_Content': text,
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})
#dataset_df = dataset_df[['cleane 'Compound','Positive', 'Neutral', 'Negative']]

# Have a look at the top 5 results.
dataset_df.head()


# In[128]:


dataset_df.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/sentiments_gutenberg.csv', header=None, index=None, sep=';', mode='a')


# In[130]:


pip install wordcloud


# In[142]:


dataset_df.head()


# In[ ]:





# In[ ]:





# In[136]:


cwd = os.getcwd()
#testing with codecs to open a html file, not necessarily needed though.
import codecs
data1 = codecs.open('/Users/sidraaziz/PycharmProjects/MLProject/Gutenberg_19th_century_English_Fiction/pg11CarolAlice-content.html','r')
#print(data1.read())

dataset1 = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/master996.csv', sep=';', engine='python')


# In[137]:


dataset1.head()


# In[143]:


dataset_df['BookName']= dataset1['Book_Name']


# In[ ]:





# In[166]:


#score = dataset_df.groupby(['Neutral', 'BookName'])


# In[ ]:





# In[167]:


#score.mean().sort_values(by='Neutral',ascending=False)


# In[175]:




lemmatized_data = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/MLProject/lemmatized_content.csv')

lemmatized_data.head()


# In[284]:


dataset_df.head()


# In[260]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[275]:


dataset_df1.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/sorted_sentiments_gutenberg.csv', header=None, index=None, sep=';', mode='a')


# In[290]:


#dataset_df['Compound', 'Positive', 'Neutral', 'Negative', 'BookName'].to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/sentiments_gutenberg.csv', index=None, sep=';', mode='a')


dataset_df.to_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/sentiments_gutenberg.csv', index=None, sep=';', mode='a')


# In[286]:


dataset_df.head()


# In[ ]:




