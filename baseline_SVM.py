import os
import numpy as np
import pandas as pd
import time
import sys
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def  read_single_data(input_file, ignore_brand):
    data = {}
    csv_read = pd.read_csv(input_file)
    rows = csv_read.values.tolist()
    if ignore_brand == True:
        rows = list(filter(lambda x: x[2] != 'brand', rows))
    random.shuffle(rows)
    train_size = int(len(rows)  * 0.6)

    train_set = rows[1:train_size]
    test_set = rows[train_size + 1: ]
    
    words = []
    labels = []
    train_data = {}
    for row in train_set:
        words.append(row[1])
        labels.append(row[2])
    train_data['data'] = words
    train_data['target'] = labels
    data['training_data'] = train_data

    words = []
    labels = []
    test_data = {}
    for row in test_set:
        words.append(row[1])
        labels.append(row[2])
    test_data['data'] = words
    test_data['target'] = labels
    data['testing_data'] = test_data    
    return data
    
    
#def read_both_data(f1, f2):
    
    
def main():
    

    
    mode = 2
    f1 = 'processed_data/all_tweets.csv'
    f2 = 'processed_data/all_descriptions.csv'
    should_ignore_brand = True

    if mode == 1:
        data =read_single_data(f1,should_ignore_brand)
    elif mode == 2:
        data =read_single_data(f2,should_ignore_brand)
    elif mode == 3:
        data = read_both_data(f1,f2,should_ignore_brand)

    ''' mode:
            1 : tweet only
            2: desciption only
            3: tweet + desciption 
    '''

    # Create feature vectors
    #minCOunt = 1
    # maxRation 0.8
    vectorizer = TfidfVectorizer(min_df=4, max_df = 0.9, sublinear_tf=True)
    
    train_vectors = vectorizer.fit_transform(data['training_data']['data'])
    test_vectors = vectorizer.transform(data['testing_data']['data'])
    train_labels = data['training_data']['target']
    test_labels = data['testing_data']['target']


    for i in range(0,10):
        print data['training_data']['target'][i]
    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print accuracy_score(test_labels,prediction_rbf)
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print accuracy_score(test_labels,prediction_linear)
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    print accuracy_score(test_labels,prediction_liblinear)



if __name__=="__main__":
    main()
