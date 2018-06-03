#!/usr/bin/python
# -*- coding: utf-8 -*-


import time

import pandas as pd

import nltk

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from TurkishStemmer import TurkishStemmer


names = [
         "Nearest Neighbors",
         "Linear SVM", "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         # "QDA"
         ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    MultinomialNB(),
    # QuadraticDiscriminantAnalysis()
]

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.min() + (cm.max()-cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



df=pd.read_csv("equal_num_sent_combined-train.txt",sep='\t',names=['liked','id','text'],engine='python',nrows=7493)

stopwords=stopwords.words('turkish')
vectorizer=TfidfVectorizer(tokenizer=tokenize,use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopwords)

y=df.liked
X=vectorizer.fit_transform(df.text)

# X = StandardScaler(with_mean=False).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)


for name, clf in zip(names, classifiers):
    time_begin = time.time()

    try:
        clf.fit(X_train, y_train)
        print(name)
    
        pred = clf.predict(X_test)
    
        confusion_mtx = confusion_matrix(y_test, pred)
        plot_confusion_matrix(confusion_mtx, classes = [-1,0,1])
    

    except TypeError as e:
        print (name, e)#, "performance in ", time.time() - time_begin, "seconds"
