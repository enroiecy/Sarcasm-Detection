import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

import re
import seaborn as sns

tweets = pd.read_csv('sarcasm_dataset_cleaned.csv')
tweets['tweet'] = tweets['tweet'].str.replace('@USER', '')
'''
sarcastic_tweets = tweets.loc[tweets['label']==0]
not_sarcastic_tweets = tweets.loc[tweets['label']==1]

stop=set(stopwords.words('english'))
corpus = []
new = tweets['tweet'].str.split()
new = new.values.tolist()
corpus = [word for tweet_words in new for word in tweet_words]
'''
#Cleaning Text
regex = re.compile('[^a-z\s]')

tweets['tweet'] = tweets['tweet'].str.lower()
tweets['tweet'] = tweets['tweet'].apply(lambda x: regex.sub('', x))
tweets['tweet'] = tweets['tweet'].str.replace('rt|http', '', regex = True)
tweets['tweet'] = tweets['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

#Tokenizing and removing stop words
tweets['tweet'] = tweets['tweet'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
tweets['tweet'] = tweets['tweet'].apply(lambda x: [word for word in x if word not in stop_words])

#Stemming
stemmer = SnowballStemmer('english')
tweets['tweet'] = tweets['tweet'].apply(lambda x: [stemmer.stem(w) for w in x])
# Joining the words back into a single text
tweets['tweet'] = tweets['tweet'].apply(lambda x: ' '.join(x))


all_words = ' '.join(word for word in tweets['tweet'])

wordcloud = WordCloud(width = 800, height = 500, background_color = 'white', 
                min_font_size = 10).generate(all_words)

plt.figure(figsize = (10, 6), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.show()