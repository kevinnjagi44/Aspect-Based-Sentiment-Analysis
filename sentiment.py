# -*- coding: utf-8 -*-
"""
Created on Thu Nov 1  6 00:35:26 2018

@author: Dhananjay
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import scipy

warnings.filterwarnings("ignore")
re_tokenize = RegexpTokenizer("[\w']+")
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
in_data_file = "data-2_train.csv"
test_data_file = "Data-2_test.csv"


#Load the data file
def data_load(in_data_file):
    data = pd.read_csv(in_data_file, skipinitialspace=True)
    dataTest = pd.read_csv(test_data_file, skipinitialspace=True)
	
    '''PreProcessing'''
    data = data_preprocess(data)
    dataTest = data_preprocess(dataTest)
    print("Data loaded and preprocessed")
	
    d_y = data["class"]

    '''Feature Extraction'''
    x_vect_train, x_vect_test = vect_ngram_data(data[['text','aspect_term']], dataTest[['text','aspect_term']])
    x_vect_train = scipy.sparse.csr_matrix(x_vect_train)
    x_vect_train = calc_adj_feature(x_vect_train, data)
    x_vect_test = scipy.sparse.csr_matrix(x_vect_test)
    x_vect_test = calc_adj_feature(x_vect_test, dataTest)    
    
    print("Features extracted")
	
    '''Train & Predict from ML model'''
    svm(x_vect_train, d_y)
    naive_bayes(x_vect_train, d_y)
    decision_tree(x_vect_train, d_y)
    
    #final model
    preds = finalSVM(x_vect_train, x_vect_test,d_y)
    
    print("Model trained and class predicted")
	
	#Write Output
    f=open("Dhananjay_Gupta_Karan_Kadakia_Data-2.txt","w+")
    for i in range(len(preds)):
        strRec = dataTest['example_id'][i] + ";;" + str(preds[i]) + "\n"
        f.write(strRec)
    f.close()

def data_preprocess(data):
    data.text = data.text.str.replace("\[comma\]", ",")
    data.aspect_term = data.aspect_term.str.replace("\[comma\]", ",")
    data.text = data.text.str.replace("_", "")
    data.aspect_term = data.aspect_term.str.replace("_", "")
    data.text = data.text.str.replace(" '", " ")
    data.aspect_term = data.aspect_term.str.replace(" '", " ")
    data.text = data.text.str.replace("' ", " ")
    data.aspect_term = data.aspect_term.str.replace("' ", " ")
    data.text = data.text.str.replace("'s", " ")
    data.aspect_term = data.aspect_term.str.replace("'s", " ")
    data.text = data.text.apply(lambda row: re.sub("'$", " ",row))
    data.aspect_term = data.aspect_term.apply(lambda row: re.sub("'$", " ",row))
    data.text = data.text.apply(lambda row: re.sub("^'", " ",row))
    data.aspect_term = data.aspect_term.apply(lambda row: re.sub("^'", " ",row))
    data["text"] = data["text"].str.lower()
    data["aspect_term"] = data["aspect_term"].str.lower()
    data = lemmatization(data)
    return data


def apply_chi2(x, y):
    chi2_selector = SelectKBest(chi2, k=4000)
    X_kbest = chi2_selector.fit_transform(x, y)
    return X_kbest

def vect_ngram_data(txt_asp_data, txt_asp_data_test): 
    textList = []
    for id,row in txt_asp_data.iterrows():
        textWords = do_re_tokenize(row['text'])
        try:
            aspIndex = textWords.index(do_re_tokenize(row['aspect_term'])[0])
            xtrWords = 5
            startIndex = max(aspIndex - xtrWords,0)
            endIndex = min(aspIndex + xtrWords,len(textWords))
            textList.append(' '.join(textWords[startIndex:endIndex+1]))
        except:
            textList.append(row['text'])
    
    textListTest = []
    for id,row in txt_asp_data_test.iterrows():
        textWords = do_re_tokenize(row['text'])
        try:
            aspIndex = textWords.index(do_re_tokenize(row['aspect_term'])[0])
            xtrWords = 5
            startIndex = max(aspIndex - xtrWords,0)
            endIndex = min(aspIndex + xtrWords,len(textWords))
            textListTest.append(' '.join(textWords[startIndex:endIndex+1]))
        except:
            textListTest.append(row['text'])
    
	
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    x_vec_train = tfidf.fit_transform(textList)
    x_vec_test = tfidf.transform(textListTest)
    return x_vec_train, x_vec_test

#Final model SVM
def finalSVM(X_train, X_test, Y_train):
    svc = LinearSVC(dual=False)
    svc.fit(X_train, Y_train)
    preds = svc.predict(X_test)
    return preds
	
def svm(x, y):
    svc = LinearSVC(dual=False)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        accuracy_list.append(svc.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0, 2])

        tp_pos = cm[0][0]


        tp_neg = cm[2][2]

        tp_neutral = cm[1][1]

        prec_pos = tp_pos / (cm[0][0] + cm[1][0] + cm[2][0])
        prec_neg = tp_neg / (cm[0][2] + cm[1][2] + cm[2][2])
        #print(tp_neg,cm[0][2],cm[1][2],cm[2][2])
        prec_neutral = tp_neutral / (cm[0][1] + cm[1][1] + cm[2][1])
        rec_pos = tp_pos / (cm[0][0] + cm[0][1] + cm[0][2])
        rec_neg = tp_neg / (cm[2][0] + cm[2][1] + cm[2][2])
        rec_neutral = tp_neutral / (cm[1][0] + cm[1][1] + cm[1][2])
        precision_list_pos.append(prec_pos)
        precision_list_neg.append(prec_neg)
        precision_list_neutral.append(prec_neutral)
        recall_list_pos.append(rec_pos)
        recall_list_neg.append(rec_neg)
        recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    print("Precision for svc (class 1) is : " + str(precision_pos))
    print("Precision for svc (class -1) is : " + str(precision_neg))
    print("Precision for svc (class 0) is : " + str(precision_neutral))
    print("Recall for svc (class 1) is : " + str(recall_pos))
    print("Recall for svc (class -1) is : " + str(recall_neg))
    print("Recall for svc (class 0) is : " + str(recall_neutral))
    print("Accuracy for svc is : " + str(accuracy))


def calc_adj_feature(x_vect, myData):
    text_file = open("pos_words.txt", "r")
    pos_words = text_file.read().split('\n')
    text_file = open("neg_words.txt", "r")
    neg_words = text_file.read().split('\n')

    conjunct_list = ['but', 'nor', 'yet', 'although', 'before', 'if', 'though', 'till', 'unless', 'until', 'what',
                     'whether', 'while','.']

    adj_class = []
    adj_dist = []
    nearest_adj = []
    pos_count = []
    neg_count = []
    sub_sent_counts = []

    for id, row in myData.iterrows():
        min_dist = len(row['text'])

        my_text = row['text']
        my_text2 = my_text

        for word in do_re_tokenize(my_text2):
            if word in conjunct_list:
                my_text2 = my_text2.replace(word, "~")

        sub_sent_count = len(my_text2.split('~'))

        for subText in my_text2.split('~'):
            if row['aspect_term'] in subText:

                my_text = subText
                break

        try:
            term_start = my_text.index(row['aspect_term'])  
            term_end = term_start + len(row['aspect_term'])  # int(re.split('--',row['term_location'])[1])
        except:
            term_start = int(re.split('--', row['term_location'])[0])
            term_end = int(re.split('--', row['term_location'])[1])
        
        negCount = 0
        posCount = 0
        for str in do_re_tokenize(my_text):

            if str in pos_words:
                posCount = posCount + 1
            elif str in neg_words:
                negCount = negCount + 1
            try:
                if term_start <= my_text.index(str) < term_end:
                    continue
            except:
                print()


            try:
                dist = abs(my_text.index(str) - my_text.index(row['aspect_term']))
            except ValueError:

                adjClass = 0
                min_dist = len(my_text)
                near_adj = str
                break

            if dist < min_dist:
                near_adj = str
                if str in pos_words:
                    adjClass = +1  
                    min_dist = dist
                elif str in neg_words:
                    adjClass = 2  
                    min_dist = dist
                else:
                    adjClass = 0  
        adj_class.append(adjClass)
        nearest_adj.append(near_adj)
        adj_dist.append(min_dist)
        pos_count.append(posCount)
        neg_count.append(negCount)
        sub_sent_counts.append(sub_sent_count)

    myData["adj_class"] = adj_class
    myData["adj_dist"] = adj_dist
    myData["pos_count"] = pos_count
    myData["neg_count"] = neg_count
    myData["sub_sent_counts"] = sub_sent_counts
    
    myData2 = myData[["adj_class", "adj_dist", "pos_count", "neg_count"]]
    myData2 = scipy.sparse.csr_matrix(myData2)

	
    return scipy.sparse.hstack((x_vect,myData2)).tocsr()


def decision_tree(x, y):
    dtree = DecisionTreeClassifier(max_depth=3)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree.fit(x_train, y_train)
        y_pred = dtree.predict(x_test)
        accuracy_list.append(dtree.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        tp_pos = cm[0][0]
        # tn = cm[2][2]
        fp = cm[0][2]
        fn = cm[2][0]
        
        tp_neg = cm[2][2]
        
        tp_neutral = cm[1][1]
        
        prec_pos = tp_pos/(cm[0][0] + cm[1][0] + cm[2][0])
        prec_neg = tp_neg/(cm[0][2] + cm[1][2] + cm[2][2])
        prec_neutral = tp_neutral/(cm[0][1] + cm[1][1] + cm[2][1])
        rec_pos = tp_pos/(cm[0][0] + cm[0][1] + cm[0][2])
        rec_neg = tp_neg/(cm[2][0] + cm[2][1] + cm[2][2])
        rec_neutral = tp_neutral/(cm[1][0] + cm[1][1] + cm[1][2])
        precision_list_pos.append(prec_pos)
        precision_list_neg.append(prec_neg)
        precision_list_neutral.append(prec_neutral)
        recall_list_pos.append(rec_pos)
        recall_list_neg.append(rec_neg)
        recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    print("Accuracy for dtree is : " + str(accuracy))
    print("Precision for dtree (class 1) is : " + str(precision_pos))
    print("Precision for dtree (class -1) is : " + str(precision_neg))
    print("Precision for dtree (class 0) is : " + str(precision_neutral))
    print("Recall for dtree (class 1) is : " + str(recall_pos))
    print("Recall for dtree (class -1) is : " + str(recall_neg))
    print("Recall for dtree (class 0) is : " + str(recall_neutral))


def naive_bayes(x, y):
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x)
    accuracy_list = []
    precision_list_pos = []
    precision_list_neg = []
    precision_list_neutral = []
    recall_list_pos = []
    recall_list_neg = []
    recall_list_neutral = []
    
    for train_index, test_index in skf.split(x, y):
        gnb = MultinomialNB()
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        accuracy_list.append(gnb.score(x_test, y_test))
        cm = confusion_matrix(y_test, y_pred)
        tp_pos = cm[0][0]
        # tn = cm[2][2]
        fp = cm[0][2]
        fn = cm[2][0]
        
        tp_neg = cm[2][2]
        
        tp_neutral = cm[1][1]
        
        prec_pos = tp_pos/(cm[0][0] + cm[1][0] + cm[2][0])
        prec_neg = tp_neg/(cm[0][2] + cm[1][2] + cm[2][2])
        # print(tp_neg,cm[0][2],cm[1][2],cm[2][2])
        prec_neutral = tp_neutral/(cm[0][1] + cm[1][1] + cm[2][1])
        rec_pos = tp_pos/(cm[0][0] + cm[0][1] + cm[0][2])
        rec_neg = tp_neg/(cm[2][0] + cm[2][1] + cm[2][2])
        rec_neutral = tp_neutral/(cm[1][0] + cm[1][1] + cm[1][2])
        precision_list_pos.append(prec_pos)
        precision_list_neg.append(prec_neg)
        precision_list_neutral.append(prec_neutral)
        recall_list_pos.append(rec_pos)
        recall_list_neg.append(rec_neg)
        recall_list_neutral.append(rec_neutral)

    accuracy = np.mean(accuracy_list)
    precision_pos = np.mean(precision_list_pos)
    precision_neg = np.mean(precision_list_neg)
    precision_neutral = np.mean(precision_list_neutral)
    recall_pos = np.mean(recall_list_pos)
    recall_neg = np.mean(recall_list_neg)
    recall_neutral = np.mean(recall_list_neutral)
    print("Accuracy for nb is : " + str(accuracy))
    print("Precision for nb (class 1) is : " + str(precision_pos))
    print("Precision for nb (class -1) is : " + str(precision_neg))
    print("Precision for nb (class 0) is : " + str(precision_neutral))
    print("Recall for nb (class 1) is : " + str(recall_pos))
    print("Recall for nb (class -1) is : " + str(recall_neg))
    print("Recall for nb (class 0) is : " + str(recall_neutral))


def tokenize(cp_in_data):
    cp_in_data["text"] = cp_in_data["text"].apply(lambda row: word_tokenize(row))
    return cp_in_data


def do_re_tokenize(row):
    x = re_tokenize.tokenize(row)
    return x


def remove_tags(row):
    row = str(row)
    cleanr = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanr, ' ', row)
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', ' ', cleantext)
    cleantext = re.sub('[^\sa-zA-Z]+', '', cleantext)
    cleantext = re.sub('\s+', ' ', cleantext)
    cleantext = cleantext[0:].strip()
    return cleantext


def lemmatization(data_stem):
    data_stem["text"] = data_stem["text"].apply(do_re_tokenize)
    data_stem["text"] = data_stem["text"].apply(lambda x: [wnl.lemmatize(y) for y in x])
    data_stem["text"] = data_stem["text"].apply(lambda x: " ".join(x))
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(do_re_tokenize)
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(lambda x: [wnl.lemmatize(y) for y in x])
    data_stem["aspect_term"] = data_stem["aspect_term"].apply(lambda x: " ".join(x))

    return data_stem


data_load(in_data_file)
