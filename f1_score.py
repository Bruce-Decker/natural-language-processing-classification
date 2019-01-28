#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:49:27 2018

@author: brucedecker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:12:18 2018

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
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

stopwords_set = set(stopwords.words("english"))
np.set_printoptions(threshold='nan')
df = pd.read_csv('train.dat.txt', sep='\t', skip_blank_lines=False)
df = df.replace(np.nan, '', regex=True)
document = []
print(df.shape[0])

words = stopwords.words("english")
stemmer = SnowballStemmer('english')




for i in range(0, df.shape[0]):
    
    new_text = re.sub('[^a-zA-Z]', ' ', df['Text'][i])
    new_text = re.sub(r"<br />", " ", new_text)
    new_text = re.sub(r"   ", " ", new_text) 
    new_text = re.sub(r"  ", " ", new_text)
    new_text = new_text.lower().split()
    #new_text = new_text.split()
    new_text = [stemmer.stem(word) for word in new_text if not word in stopwords_set]
    new_text = ' '.join(new_text)  
    document.append(new_text)
    
df['cleaned'] = df['Text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwords_set]).lower())

cv = TfidfVectorizer(sublinear_tf=True, min_df=7, stop_words='english')

#ch2 = SelectKBest(chi2, k=2000)
ch2 = SelectPercentile(chi2, percentile=9.5)
X = cv.fit_transform(df['cleaned'])

y = df.iloc[:, 0].values
  
feature_names = cv.get_feature_names()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
#selected_feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]

clf = LinearSVC(C=1.0, penalty='l1', max_iter=3000,  dual=False)
#clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='weighted'))

