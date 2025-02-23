# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:14:01 2024

@author: Jin Li
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

# Vocabulary sizes to test
vocab_sizes = [10, 100, 1000, 10000]
accuracies_no_stop_words = []
accuracies_with_stop_words = []
accuracies_tfidf = []

for vocab_size in vocab_sizes:
    
    # Without stop words
    vectorizer = CountVectorizer(max_features=vocab_size)
    print("vocab_size:", vocab_size)
    print("vocab_sizes:", vocab_sizes)
    print("vectorizer",vectorizer)
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
    
    # Train and evaluate the classifier
    clf = KNeighborsClassifier()
    clf.fit(X_train, data_train.target)
    pred = clf.predict(X_test)
    accuracies_no_stop_words.append(accuracy_score(data_test.target, pred))
    
    # With stop words
    vectorizer_sw = CountVectorizer(max_features=vocab_size, stop_words='english')
    X_train_sw = vectorizer_sw.fit_transform(data_train.data)
    X_test_sw = vectorizer_sw.transform(data_test.data)
    
    # Train and evaluate the classifier
    clf.fit(X_train_sw, data_train.target)
    pred_sw = clf.predict(X_test_sw)
    accuracies_with_stop_words.append(accuracy_score(data_test.target, pred_sw))
    
    # With TF-IDF weighting
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_sw)
    X_test_tfidf = tfidf_transformer.transform(X_test_sw)
    
    # Train and evaluate the classifier
    clf.fit(X_train_tfidf, data_train.target)
    pred_tfidf = clf.predict(X_test_tfidf)
    accuracies_tfidf.append(accuracy_score(data_test.target, pred_tfidf))


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(vocab_sizes, accuracies_no_stop_words, label='No Stop Words', marker='o')
plt.plot(vocab_sizes, accuracies_with_stop_words, label='With Stop Words', marker='s')
plt.plot(vocab_sizes, accuracies_tfidf, label='TF-IDF + Stop Words', marker='^')
plt.axhline(y=accuracy_score(data_test.target, clf.predict(X_test)), color='r', linestyle='-', label='Full Vocabulary')
plt.xscale('log')
plt.xlabel('Vocabulary Size')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy vs Vocabulary Size')
plt.legend()
plt.grid(True)
plt.show()

#%%



