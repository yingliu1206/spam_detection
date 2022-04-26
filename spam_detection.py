#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd 
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
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

def preprocess(messages_bow, dataframe):
            
    # tfidf
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    print('TF-IDF shape:', messages_tfidf.shape)
    
    # add feature 1
    lb = preprocessing.LabelBinarizer()
    feature1 = lb.fit_transform(dataframe['has_url'])
    tfidf_feature1 = sc.sparse.hstack((messages_tfidf, feature1))
    
    # add feature 2
    feature2 = lb.fit_transform(dataframe['has_phoneNum'])
    tfidf_feature2 = sc.sparse.hstack((messages_tfidf, feature2))
    
    # add feature 3
    feature3 = lb.fit_transform(dataframe['has_currency'])
    tfidf_feature3 = sc.sparse.hstack((messages_tfidf, feature3))
    
    # all in
    tfidf_all = sc.sparse.hstack((messages_tfidf, feature1, feature2, feature3))
    
    # train models with different features input
    prepared_features = {"base features": messages_tfidf, 
                         "base + has_urls": tfidf_feature1,
                         "base + has_phone_number": tfidf_feature2,
                         "base + has_currency_symbol": tfidf_feature3,
                         "all_in": tfidf_all}
    
    return prepared_features
    
def main(dataset):
    
    message = pd.read_csv(dataset, sep='\t', names=["labels","message"])
    print(f"The overview of the dataset: {message.head()}")
    
    # Exploratory Data Analysis
    message.describe()
    
    message.groupby('labels').describe()
    
    # add a column length
    message['length'] = message['message'].apply(len)
    
    # Data Visualization    
    message['length'].plot(bins=50,kind='hist', title = 'The distribution of length of messages', xlabel = 'length')
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
    
    # train-test split
    msg_train,msg_test,label_train,label_test = train_test_split(message, message['labels'], test_size=0.3)
    print("The length of training set:", len(msg_train))
    print("The length of training labels:", len(label_train))
    print("The length of test set:", len(msg_test))
    print("The length of test labels:", len(label_test))
    
    # vectorization
    bow_transformer = CountVectorizer(analyzer=text_process)
    messages_bow_train = bow_transformer.fit_transform(msg_train['message'])
    print('Shape of Sparse Matrix of training set: ',messages_bow_train.shape)
    
    messages_bow_test = bow_transformer.transform(msg_test['message'])
    print('Shape of Sparse Matrix of test set: ',messages_bow_test.shape)    
    
    # preprocessing
    prepared_features_train = preprocess(messages_bow_train, msg_train)
    prepared_features_test = preprocess(messages_bow_test, msg_test)
    
    for cond, features in prepared_features_train.items():
        print(f"Features: {cond}")
        spam_detect_model = MultinomialNB().fit(features, label_train)   
        all_predictions = spam_detect_model.predict(prepared_features_test[cond])  
        
        print(classification_report(label_test,all_predictions))
        print(confusion_matrix(label_test,all_predictions))

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SMSSpamCollection",
                        help="tab separated dataset")

    args = parser.parse_args()

    main(args.dataset)