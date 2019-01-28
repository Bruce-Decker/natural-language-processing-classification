#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:55:31 2018

@author: brucedecker
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import nltk
nltk.download('punkt')
import sklearn
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile




stemmer = SnowballStemmer('english')

stopwords_set = set(stopwords.words("english"))
np.set_printoptions(threshold='nan')
data_frame = pd.read_csv('train.dat.txt', sep='\t', keep_default_na=False, skip_blank_lines=False)
df_x_test = pd.read_csv('test.dat.txt', sep='\t',  names = ["Text"], engine='python', keep_default_na=False, skip_blank_lines=False)
df_x_test = df_x_test.replace(np.nan, '', regex=True)
row_test = pd.read_csv('format.dat.txt', sep='\t')
data_frame = data_frame.replace(np.nan, '', regex=True)
document = []
document_test = []
print(data_frame.shape[0])

for i in range(0, data_frame.shape[0]):
    new_text = ' '.join([stemmer.stem(word) for word in re.sub('[^a-zA-Z]', ' ', data_frame['Text'][i]).split() if not word in stopwords_set]).lower()
    document.append(new_text)

for i in range(0, df_x_test.shape[0]):
    new_text_2 = ' '.join([stemmer.stem(word) for word in re.sub('[^a-zA-Z]', ' ', df_x_test['Text'][i]).split() if not word in stopwords_set]).lower()
    document_test.append(new_text_2)

df_x_test['removed_test'] = df_x_test['Text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
tfid_v = TfidfVectorizer(sublinear_tf=True, min_df=6, stop_words='english')
#select_features = SelectKBest(chi2, k=2000)
select_features = SelectPercentile(chi2, percentile=11.5)
X = tfid_v.fit_transform(document).toarray()
X_test = tfid_v.transform(document_test).toarray()
y = data_frame.iloc[:, 0].values   
X = select_features.fit_transform(X, y)
X_test = select_features.transform(X_test)
classifier = LinearSVC(C=1.0, penalty='l1', max_iter=4500, dual=False)
classifier.fit(X, y)
y_pred = classifier.predict(X_test)
np.savetxt('new.txt', y_pred, delimiter=" ", fmt="%s") 
