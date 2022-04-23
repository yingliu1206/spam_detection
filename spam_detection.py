#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
import string
import scipy as sc
from nltk.corpus import stopwords
import re


# Text Pre-processing
def text_process(mess):
    
    # remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # remove stopwords
    clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    return clean_mess

# extract url from message
def has_url(line):
    '''
    has url or not
    '''

    r = re.match("www.", line)
    
    if r != None:
        return 'yes'
    else:
        return 'no'
    
    
# extract phone number from message
def has_phoneNumber(line):
    '''
    has phone number or not
    
    number formats:
    000-000-0000
    000 000 0000
    000.000.0000
    
    (000)000-0000
    (000)000 0000
    (000)000.0000
    (000) 000-0000
    (000) 000 0000
    (000) 000.0000
    
    000-0000
    000 0000
    000.0000
    0000000
    0000000000
    (000)0000000
    
    # Detect phone numbers with country code
    +00 000 000 0000
    +00.000.000.0000
    +00-000-000-0000
    +000000000000
    0000 0000000000
    0000-000-000-0000
    00000000000000
    +00 (000)000 0000
    0000 (000)000-0000
    0000(000)000-0000 
    '''

    r = re.findall('((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))', line)

    if len(r) != 0:
        return 'yes'
    else:
        return 'no'
    
# extract currency symbol from message
def has_currency_symbol(line):
    '''
    has currency symbol or not

    '''

    r = re.findall("\\$|\\£", line)

    if len(r) != 0:
        return 'yes'
    else:
        return 'no'
    
def main(dataset):
    
    # import dataset
    # to do: change to dataset
    
    message = pd.read_csv(dataset, sep='\t', names=["labels","message"])
    print(f"The overview of the dataset: {message.head()}")
    
    # Exploratory Data Analysis
    message.describe()
    
    message.groupby('labels').describe()
    
    # add a column length
    message['length'] = message['message'].apply(len)
    
    # Data Visualization    
    message['length'].plot(bins=50,kind='hist', title = 'The distribution of length of messages')
    message.length.describe()
    
    # data preprocessing
    message['message'].apply(text_process)
    
    # feature 1 - has url or not
    l_url = []
    for i in message['message']:
        l_url.append(has_url(i))
    
    message['has_url'] = l_url
    
    # feature 2 - has phone number or not
    l_phone = []
    for i in message['message']:
        l_phone.append(has_phoneNumber(i))
    
    message['has_phoneNum'] = l_phone 
    
    # feature 3 - has currency symbol or not
    l_currency = []
    for i in message['message']:
        l_currency.append(has_currency_symbol(i))
    
    message['has_currency'] = l_currency
    
        
    # vectorization
    bow_transformer = CountVectorizer(analyzer=text_process).fit(message['message'])
    messages_bow = bow_transformer.transform(message['message'])
    print('Shape of Sparse Matrix: ',messages_bow.shape)
    print('Amount of non-zero occurences:',messages_bow.nnz)
    
    # calculate the sparsity
    sparsity =(100.0 * messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
    print('sparsity:{}'.format(round(sparsity)))
    
    # tfidf
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    print('TF-IDF shape:', messages_tfidf.shape)
    
    # add feature 1
    lb = preprocessing.LabelBinarizer()
    feature1 = lb.fit_transform(l_url)
    tfidf_feature1 = sc.sparse.hstack((messages_tfidf, feature1))
    
    # add feature 2
    feature2 = lb.fit_transform(l_phone)
    tfidf_feature12 = sc.sparse.hstack((tfidf_feature1, feature2))
    
    # add feature 3
    feature3 = lb.fit_transform(l_currency)
    tfidf_feature123 = sc.sparse.hstack((tfidf_feature12, feature3))
    
    # train basic model
    
    
    spam_detect_model = MultinomialNB().fit(messages_tfidf, message['labels'])
    
    spam_detect_model = MultinomialNB().fit(tfidf_feature1, message['labels'])
    all_predictions = spam_detect_model.predict(tfidf_feature1)
    
    print(classification_report(message['labels'],all_predictions))
    print(confusion_matrix(message['labels'],all_predictions))

    



    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SMSSpamCollection",
                        help="tab separated dataset")

    args = parser.parse_args()

    main(args.dataset)