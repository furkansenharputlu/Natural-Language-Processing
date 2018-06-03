# -*- coding: utf-8 -*-
from TurkishStemmer import TurkishStemmer
from nltk.corpus import stopwords
import pandas as pd
import string


stemmer=TurkishStemmer()
stopwords=[word.encode('utf-8') for word in stopwords.words('turkish')]


df=pd.read_csv("torku_cikolata.txt",sep='\t|  ',names=['sentiments','ids','tweets'],engine='python',nrows=270)

# Remove hashtags, mentions, links, pictures
def clean_tweets(tweets):
	clean_words = []
	for tweet in tweets:
		tweet=tweet.translate(None, string.punctuation).lower()
		for word in tweet.split():
			if not word.startswith('#') \
			 and not word.startswith('@') \
			 and not word.startswith('http') \
			 and not word.startswith('RT') \
			 and not word.startswith('pictwitter') \
			 and (word!="" or word!="RT"):
				clean_words.append(word)
	return clean_words


# Remove stopwords and stem
def stem(tweets):
	clean_words=clean_tweets(tweets)
	stems=[]
	for word in clean_words:
		if word not in stopwords:
			stems.append(stemmer.stem(word))
	return stems


for stem in stem(df.tweets):
	print stem+"\n"