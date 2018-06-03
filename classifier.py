#!/usr/bin/python
# -*- coding: utf-8 -*-


from  time import time

import sys

import pandas as pd

import nltk

from collections import Counter

from nltk.corpus import stopwords

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from TurkishStemmer import TurkishStemmer


names = [
        # "QDA",
        "Nearest Neighbors",
        "LinearSVC",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes"
         ]

classifiers = [
    # QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(),
    LinearSVC(multi_class="crammer_singer"),
    LinearSVC(C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    MultinomialNB(),
]

coefficients = [.51, 0, .59, .47, 0, .59, .50, .59]
s =sum(coefficients)
coefficients = [x/s for x in coefficients]


stemmer=TurkishStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item.encode('utf-8')))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def multiple_classfier_prediction(X, y,query_tweets):
    # results = [0]*len(query_tweets)
    results = [Counter() for _ in range(len(query_tweets))]
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        time_begin = time()
        
        try:
            clf.fit(X, y)
            
            tweet_vector = vectorizer.transform(query_tweets)
            
            temp_results = clf.predict(tweet_vector)
            print(temp_results, name, 'in', time() - time_begin, 'seconds')
            
            for j in range(len(results)):
                results[j][temp_results[j]] += coefficients[i]
                # results[j] += temp_results[j] * coefficients[i]
                
        
        except TypeError as e:
            print(name, e)
        print
            
    return results



if __name__ == '__main__':
    
    
    try:
        train_file = sys.argv[1]
        inputtxt = sys.argv[2]
        outputtxt = sys.argv[3]
    except:
        train_file = "combined-train.txt"
        inputtxt = "input.txt"
        outputtxt = 'output.txt'

    with open(train_file, "r") as ts:
        lines = ts.readlines()
        _nrows = len(lines)
    
    df=pd.read_csv(train_file,sep='\t',names=['liked','id','text'],engine='python',nrows=_nrows)
    
    stopwords=stopwords.words('turkish')
    vectorizer=TfidfVectorizer(tokenizer=tokenize,use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopwords)
    
    y=df.liked
    X=vectorizer.fit_transform(df.text)
    
    
    with open(inputtxt, "r") as input_file:
        query_tweets = input_file.readlines()

    results = multiple_classfier_prediction(X, y,query_tweets)
    
    print("results: ")
    # for res in results:
    #     print(res.most_common(3))

    
    with open(outputtxt,'w') as output_file:
        for result, query_tweet in  zip(results,query_tweets):
            output_file.write(str(result.most_common(1)[0][0]) + '\t\t\t' + query_tweet)
            
    
    
