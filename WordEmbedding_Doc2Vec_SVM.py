import os
import numpy as np
import pandas as pd
from gensim import utils
from random import shuffle
import time
import gensim as ge
from gensim.models.doc2vec import LabeledSentence
import os.path
import sys
import mysvm
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# for fairness in comparison, the training set and testing set should be the same for both baseline and Doc2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

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
    
    
#def read_both_data(f1


def create_Doc2Vec_input(data, isTrained, data_info) :
    if isTrained == True:
        f_male = open('Doc2Vec_data/train_male.txt','w')
        f_female = open('Doc2Vec_data/train_female.txt','w')
        f_brand  = open('Doc2Vec_data/train_brand.txt','w')
    else:
        f_male = open('Doc2Vec_data/test_male.txt','w')
        f_female = open('Doc2Vec_data/test_female.txt','w')
        f_brand  = open('Doc2Vec_data/test_brand.txt','w')
    for  i in range(0, len(data['data'])):
        if data['target'][i] == 'male':
            f_male.write(data['data'][i] + '\n')
            if isTrained == True:
                data_info['TRAIN_MALE_SIZE']  = data_info['TRAIN_MALE_SIZE'] + 1
            else:
                data_info['TEST_MALE_SIZE'] = data_info['TEST_MALE_SIZE'] + 1
        elif data['target'][i] == 'female':
            f_female.write(data['data'][i] + '\n')
            if isTrained == True:
                data_info['TRAIN_FEMALE_SIZE'] = data_info['TRAIN_FEMALE_SIZE']+ 1
            else:
                data_info['TEST_FEMALE_SIZE'] = data_info['TEST_FEMALE_SIZE'] + 1
        else:
            f_brand.write(data['data'][i] + '\n')
            if isTrained == True:
                data_info['TRAIN_BRAND_SIZE'] = data_info['TRAIN_BRAND_SIZE'] + 1
            else:
                data_info['TEST_BRAND_SIZE'] = data_info['TEST_BRAND_SIZE'] + 1
    
    f_male.close()
    f_female.close()
    f_brand.close()
    return data_info

def get_word_vectors(model, prefix, size, label):
    vectors = []
    labels = []
    for i in range(size):
        vectors.append(model.docvecs[prefix + str(i)])
        labels.append(label)
    return vectors, labels

    
def main():

    mode = 1
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

        

    data_info = {'TRAIN_MALE_SIZE':0, 'TRAIN_FEMALE_SIZE':0, 'TRAIN_BRAND_SIZE':0, 'TEST_MALE_SIZE':0,'TEST_FEMALE_SIZE':0,'TEST_BRAND_SIZE':0}

    # Create feature vectors
    #minCOunt = 1
    # maxRation 0.8
    vectorizer = TfidfVectorizer(min_df=4, max_df = 0.9, sublinear_tf=True)
    
    train_vectors = vectorizer.fit_transform(data['training_data']['data'])
    test_vectors = vectorizer.transform(data['testing_data']['data'])
    train_labels = data['training_data']['target']
    test_labels = data['testing_data']['target']


    # save to files required for Doc2Vec
    data_info = create_Doc2Vec_input(data['training_data'], True, data_info)
    data_info = create_Doc2Vec_input(data['testing_data'], False, data_info)

    #this is for modeling whole test instances as well
    # for testing new upcoming test cases-> inference test with clustering similarity

    if should_ignore_brand == True:
        sources = {'Doc2Vec_data/train_male.txt':'TRAIN_MALE', 'Doc2Vec_data/train_female.txt':'TRAIN_FEMALE',
               'Doc2Vec_data/test_male.txt':'TEST_MALE', 'Doc2Vec_data/test_female.txt':'TEST_FEMALE'}
    else:
        sources = {'Doc2Vec_data/train_male.txt':'TRAIN_MALE', 'Doc2Vec_data/train_female.txt':'TRAIN_FEMALE',
                   'Doc2Vec_data/train_brand.txt':'TRAIN_BRAND', 'Doc2Vec_data/test_male.txt':'TEST_MALE',
                   'Doc2Vec_data/test_female.txt':'TEST_FEMALE','Doc2Vec_data/test_brand.txt':'TEST_BRAND'}

    #docs = ge.models.doc2vec.LabeledLineSentence(sources)
    docs = LabeledLineSentence(sources)
    #print docs.to_array()
    #print docs.to_array()[0]
    
    vector_size = 100
    window_size = 10
    min_count = 1
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 20
    dm = 0 #0 = dbow; 1 = dmpv # algorithm 0=<-- important!
    worker_count = 4

    saved_path = "processed_data/model.bin"
	
    #if os.path.isfile(saved_path) == False:
    model = ge.models.Doc2Vec(size=vector_size, hs = 0, iter = train_epoch, window=window_size, min_count=min_count, workers=worker_count, dm=dm, negative=negative_size)
    model.build_vocab(docs.to_array())
    model.train(docs.sentences_perm(), total_examples = model.corpus_count, epochs = model.iter)
    model.save(saved_path)
    model = ge.models.Doc2Vec.load(saved_path)

    print model.corpus_count
    print len(model.docvecs)
    
    # without word embedding
    print 'Running SVM with IFIDF'
    #mysvm.runSVM(train_vectors,train_labels,test_vectors, test_labels)

    # with word embedding
    train_male , train_male_label = get_word_vectors(model, 'TRAIN_MALE_', data_info['TRAIN_MALE_SIZE'], 'male')
    train_female , train_female_label = get_word_vectors(model, 'TRAIN_FEMALE_', data_info['TRAIN_FEMALE_SIZE'], 'female')
    if should_ignore_brand == True:
        train_we_vectors = train_male + train_female
        train_we_labels = train_male_label + train_female_label
    else:
        train_brand , train_brand_label = get_word_vectors(model, 'TRAIN_BRAND_', data_info['TRAIN_BRAND_SIZE'], 'brand')        
        train_we_vectors = train_male  + train_female+ train_brand
        train_we_labels = train_male_label + train_female_label + train_brand_label
    
    test_male , test_male_label = get_word_vectors(model, 'TEST_MALE_', data_info['TEST_MALE_SIZE'], 'male')
    test_female , test_female_label = get_word_vectors(model, 'TEST_FEMALE_', data_info['TEST_FEMALE_SIZE'], 'female')
    if should_ignore_brand == True:
        test_we_vectors = test_male + test_female
        test_we_labels = test_male_label + test_female_label
    else:
        test_brand , test_brand_label = get_word_vectors(model, 'TEST_BRAND_', data_info['TEST_BRAND_SIZE'], 'brand')        
        test_we_vectors = test_male  + test_female + test_brand
        test_we_labels = test_male_label + test_female_label + test_brand_label

    print '------'
    print len(test_male)
    print len(test_female)
    print np.shape(test_we_vectors)
    print len(train_male)
    print len(train_female)
    print np.shape(train_we_vectors)
    print np.shape(train_we_labels)

    t_male = open('debug_male.txt','w')
    t_female = open('debug_female.txt','w')

    for row  in train_we_labels:
        t_male.write('%s \n' %(row))
    t_male.close()
    t_female.write(test_we_labels)
    t_female.close()
    #print train_we_vectors[0]
    #print train_vectors[0]
    print 'Running SVM with Doc2vec'

    #classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    #classifier.fit(train_we_vectors,train_we_labels)

    #print classifier.score(test_we_vectors,test_we_labels)
    #mysvm.runSVM(train_we_vectors,train_we_labels,test_we_vectors, test_we_labels)
    
if __name__=="__main__":
    main()
