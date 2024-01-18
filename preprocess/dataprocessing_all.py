# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:55:49 2023

@author: xingc
"""

import os
for dirname, _, filenames in os.walk('Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
from numpy import random
import csv

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras 
from keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM,Dense,Bidirectional,Input
from keras.models import Model
#import torch
import transformers

from collections import Counter
from tokenizers import BertWordPieceTokenizer


stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

#------------------------------------------------------------------------------
# --Data formatting functions--
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
# Removing URL's
def remove_url(text):
    return re.sub(r'http\S+', '', text)
#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)
#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words

# --Data encoding function--
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=400):

    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

# --Model definition--
def build_model(transformer, max_len=400):
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

#------------------------------------------------------------------------------
# Data washing
df = pd.read_json("../data/sarcasm_detection_shared_task_twitter_training.jsonl", lines=True)
df.head()

#df.drop('context', axis=1, inplace=True)
#Apply function on review column
df['response'] = df['response'].str.replace('@USER', '')
df['response']=df['response'].apply(denoise_text)

df['label'] = df['label'].replace(to_replace="SARCASM",value="1")
df['label'] = df['label'].replace(to_replace="NOT_SARCASM",value="0")
df = df.sample(frac=1).reset_index(drop=True)

X = df['response']
y = df['label']
#------------------------------------------------------------------------------
# Data washing
df2 = pd.read_json("../data/covid-19-test-data.jsonl", lines=True)
df2.head()

#df.drop('context', axis=1, inplace=True)
#Apply function on review column
df2['text'] = df2['text'].str.replace('@USER', '')
df2['text']=df2['text'].apply(denoise_text)

df2['label'] = df2['label'].replace(to_replace="SARCASM",value="1")
df2['label'] = df2['label'].replace(to_replace="NOT_SARCASM",value="0")
df2 = df2.sample(frac=1).reset_index(drop=True)

# Training/Testing splitting
X2 = df2['text']
y2 = df2['label']
#------------------------------------------------------------------------------
df3 = pd.read_csv("../data/5g_raw_2.csv")
df3.head()

#df.drop('context', axis=1, inplace=True)
#Apply function on review column
df3['Text'] = df3['Text'].str.replace('@USER', '')
df3['Text']=df3['Text'].apply(denoise_text)
#df3 = df3.sample(frac=1).reset_index(drop=True)

X3 = df3['Text']
y3 = pd.Series(['2']*X3.size, copy=False)
#------------------------------------------------------------------------------


X_t = X.values
X_t2 = X2.values
X_t3 = X3.values
y_t = y.values
y_t2 = y2.values
y_t3 = y3.values

X_conc = np.concatenate((X_t, X_t2, X_t3))
y_conc = np.concatenate((y_t, y_t2, y_t3))
#Test_num = random.randint(0.2*X.size, 0.3*X.size)
#Train_num = X.size - Test_num

#X_tot = pd.concat([X_t, X_t2])
#y_tot = pd.concat([y_t, y_t2])

X_tot = pd.Series(X_conc, copy=False)
y_tot = pd.Series(y_conc, copy=False)

Test_train_ind = ['train']*(X.size+50)+['test']*50+['predic']*X3.size
Test_train_ser = pd.Series(Test_train_ind, copy=False)
Y = pd.concat([Test_train_ser, y_tot], axis=1)

X_tot.to_csv(r'../data/corpus/twcvd3.csv', index=False, header=False)
Y.to_csv(r'../data/twcvd3.csv', index=True, header=False)


'''
with open(input.csv,'r') as csvinput:
    with open(output.csv, 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            writer.writerow(row+['Berry'])
            '''
#X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0 , stratify = y)