import os
import numpy as np
import pandas as pd
from gensim import utils
from random import shuffle
import time
from sys import platform
from collections import Counter, defaultdict
from gensim.models.doc2vec import LabeledSentence
import os.path
import sys
import mysvm
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import shutil
import smart_open
import gensim as ge
import analysis_graph_eval as eval_graph
# for fairness in comparison, the training set and testing set should be the same for both baseline and Doc2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


#https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])
    
# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        #tfidf = TfidfVectorizer(min_df=4, max_df = 0.9, sublinear_tf=True)
        
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self
    
    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])



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
    
def prepend_line(infile, outfile, line):
    """ 
    Function use to prepend lines using bash utilities in Linux. 
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)

def prepend_slow(infile, outfile, line):
    """
    Slower way to prepend the line by re-creating the inputfile.
    """
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)

def get_lines(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_lines = sum(1 for line in f)
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims
	
def  read_single_data(input_file, ignore_brand, ratio):
    data = {}
    csv_read = pd.read_csv(input_file)
    rows = csv_read.values.tolist()
    if ignore_brand == True:
        rows = list(filter(lambda x: x[2] != 'brand', rows))
    #random.shuffle(rows)
    train_size = int(len(rows)  * ratio)

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
    
def get_glove_word_embedding(path, all_words):
    glove_data = {}
    with open(path, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0]
            nums = map(float, parts[1:])
            if word in all_words:
                glove_data[word] = np.array(nums)
    return glove_data

#def read_both_data(f1


def create_Doc2Vec_input(data, labels,  isTrained, data_info) :
    if isTrained == True:
        f_male = open('Doc2Vec_data/train_male.txt','w')
        f_female = open('Doc2Vec_data/train_female.txt','w')
        f_brand  = open('Doc2Vec_data/train_brand.txt','w')
    else:
        f_male = open('Doc2Vec_data/test_male.txt','w')
        f_female = open('Doc2Vec_data/test_female.txt','w')
        f_brand  = open('Doc2Vec_data/test_brand.txt','w')
    for  i in range(0, len(data)):
        if labels[i] == 'male':
            f_male.write(data[i] + '\n')
            if isTrained == True:
                data_info['TRAIN_MALE_SIZE']  = data_info['TRAIN_MALE_SIZE'] + 1
            else:
                data_info['TEST_MALE_SIZE'] = data_info['TEST_MALE_SIZE'] + 1
        elif labels[i] == 'female':
            f_female.write(data[i] + '\n')
            if isTrained == True:
                data_info['TRAIN_FEMALE_SIZE'] = data_info['TRAIN_FEMALE_SIZE']+ 1
            else:
                data_info['TEST_FEMALE_SIZE'] = data_info['TEST_FEMALE_SIZE'] + 1
        else:
            f_brand.write(data[i] + '\n')
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

def get_transformed_data_glove(glove_file, train, test,train_label, test_label):
    #http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    #https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb
    X = train+ test
    all_words= []
    for row in X:
        all_words.append(row.split())

    training_sentences  = all_words[0:len(train)]
    testing_sentences = all_words[len(train):]

    corpus = set(w for words in all_words for w in words)
    glove_model = get_glove_word_embedding(glove_file, corpus)
    
    glove_vectorizer = TfidfEmbeddingVectorizer(glove_model)

    glove_train_vectors = glove_vectorizer.fit(training_sentences, train_label).transform(training_sentences)
    glove_test_vectors = glove_vectorizer.transform(testing_sentences)

    return glove_train_vectors, glove_test_vectors, train_label, test_label




def get_transformed_data_Doc2vec(should_ignore_brand,train_data,train_label, test_data,test_label):
    data_info = {'TRAIN_MALE_SIZE':0, 'TRAIN_FEMALE_SIZE':0, 'TRAIN_BRAND_SIZE':0, 'TEST_MALE_SIZE':0,'TEST_FEMALE_SIZE':0,'TEST_BRAND_SIZE':0}

    # save to files required for Doc2Vec
    data_info = create_Doc2Vec_input(train_data, train_label, True, data_info)
    data_info = create_Doc2Vec_input(test_data,test_label, False, data_info)

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
    sampling_threshold = 1e-4
    negative_size = 5
    train_epoch = 15
    dm = 0 #0 = dbow; 1 = dmpv # algorithm 0=<-- important!
    worker_count = 4

    saved_path = "processed_data/glove_model.bin"

    #if os.path.isfile(saved_path) == False:
    
    model = ge.models.Doc2Vec(size=vector_size, hs = 0, window=window_size, min_count=min_count, workers=worker_count, dm=dm, negative=negative_size)

    #model=ge.models.Word2Vec.load_word2vec_format(gensim_file,binary=False) #GloVe Model https://github.com/jroakes/glove-to-word2vec/blob/master/convert.py
    model.build_vocab(docs.to_array())
    for epoche in range(train_epoch):
        model.train(docs.sentences_perm())

    #model = ge.models.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)

    

    model.save(saved_path)
    model = ge.models.Doc2Vec.load(saved_path)

    print model.corpus_count
    print len(model.docvecs)

    
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

    return train_we_vectors, test_we_vectors, train_we_labels, test_we_labels

def get_tranformed_data_tfidf(train, test, train_label,test_label):
    vectorizer = CountVectorizer()
    
    train_vectors = vectorizer.fit_transform(train)
    test_vectors = vectorizer.transform(test)

    return train_vectors, test_vectors, train_label, test_label,

def get_tranformed_data_pca_tfidf(train, test, train_label,test_label):
    vectorizer = CountVectorizer()
    
    train_vectors = vectorizer.fit_transform(train)
    test_vectors = vectorizer.transform(test)

    pca = PCA(n_components = 100)
    train_transformed = pca.fit_transform(train_vectors.toarray())

    test_transformed = pca.transform(test_vectors.toarray())
    
    return train_transformed, test_transformed, train_label, test_label,

def main():

    mode = 1
    f1 = 'processed_data/all_tweets.csv'
    f2 = 'processed_data/all_descriptions.csv'
    should_ignore_brand = True
    ratio = 0.6

    kfold = 10
    kfold_acc =[]
    kfold_prec=[]
    kfold_recall = []
    kfold_f1 = []

    # data is not shuffled like infer test for crossvalidaiton purposes
    if mode == 1:
        data =read_single_data(f1,should_ignore_brand,ratio)
    elif mode == 2:
        data =read_single_data(f2,should_ignore_brand, ratio)
    elif mode == 3:
        data = read_both_data(f1,f2,should_ignore_brand, ratio)

    all_data =  data['training_data']['data'] + data['testing_data']['data']
    all_label = data['training_data']['target'] + data['testing_data']['target']
        
    results = open('results2.txt','a')
    for k in range(0, kfold):
        


        
        train_data = [x for i, x in enumerate(all_data) if i % kfold != k]
        train_labels = [x for i, x in enumerate(all_label) if i % kfold != k]
        
        test_data = [x for i, x in enumerate(all_data) if i % kfold == k]
        test_labels = [x for i, x in enumerate(all_label) if i % kfold == k]

        
        ''' mode:
                1 : tweet only
                2: desciption only
                3: tweet + desciption 
        '''
        
        
        # This is for Glove
        glove_file="pretrained_data/glove.twitter.27B.100d.txt"

        
        print 'Training GloVe'
        glove_train_vectors, glove_test_vectors, glove_train_labels, glove_test_labels = get_transformed_data_glove(glove_file, train_data, test_data , train_labels, test_labels)
        print 'Done GloVe'
        print np.shape(glove_train_vectors)
        print np.shape(glove_test_vectors)
        
        
        print 'Training IFIDF'
        ifidf_train_vectors, ifidf_test_vectors, ifidf_train_labels, ifidf_test_labels = get_tranformed_data_tfidf(train_data, test_data , train_labels, test_labels)
        print 'Done IFIDF'
        print np.shape(ifidf_train_vectors)
        print np.shape(ifidf_test_vectors)

        print 'Training IFIDF + PCA'
        ifidf_pca_train_vectors, ifidf_pca_test_vectors, ifidf_pca_train_labels, ifidf_pca_test_labels = get_tranformed_data_pca_tfidf(train_data,  test_data ,train_labels, test_labels)
        print 'Done IFIDF'
        print np.shape(ifidf_pca_train_vectors)
        print np.shape(ifidf_pca_test_vectors)
        
        print 'Training Doc2vec'
        doc2vec_train_vectors, doc2vec_test_vectors, doc2vec_train_labels, doc2vec_test_labels = get_transformed_data_Doc2vec(should_ignore_brand,  train_data , train_labels, test_data , test_labels)
        print 'Done Doc2vec'
        print np.shape(doc2vec_train_vectors)
        print np.shape(doc2vec_test_vectors)
        

        acc_ifidf = []
        acc_glove = []
        acc_doc2vec = []
        acc_pca_ifidf  =[]

        prec_ifidf = []
        prec_glove = []
        prec_doc2vec = []
        prec_pca_ifidf  =[]

        recall_ifidf = []
        recall_glove = []
        recall_doc2vec = []
        recall_pca_ifidf  =[]

        f1_ifidf = []
        f1_glove = []
        f1_doc2vec = []
        f1_pca_ifidf  =[]
        
        # without word embedding
        print 'Running SVM with IFIDF'
        a,b,c,d = mysvm.runSVM(ifidf_train_vectors,ifidf_train_labels,ifidf_test_vectors, ifidf_test_labels)
        acc_ifidf.append(a)
        prec_ifidf.append(b)
        recall_ifidf.append(c)
        f1_ifidf.append(d)

        print 'Running SVM with IFIDF + PCA'
        a0 ,b0 ,c0,d0= mysvm.runSVM(ifidf_pca_train_vectors,ifidf_pca_train_labels,ifidf_pca_test_vectors, ifidf_pca_test_labels)
        acc_pca_ifidf.append(a0)
        prec_pca_ifidf.append(b0)
        recall_pca_ifidf.append(c0)
        f1_pca_ifidf.append(d0)

        
        print 'Running SVM with IFIDF + glove'
        a1,b1,c1,d1 = mysvm.runSVM(glove_train_vectors,glove_train_labels,glove_test_vectors, glove_test_labels)
        acc_glove.append(a1)
        prec_glove.append(b1)
        recall_glove.append(c1)
        f1_glove.append(d1)

        
        
        print 'Running SVM with Doc2vec'
        a2,b2 ,c2,d2= mysvm.runSVM(doc2vec_train_vectors,doc2vec_train_labels,doc2vec_test_vectors, doc2vec_test_labels)
        acc_doc2vec.append(a2)
        prec_doc2vec.append(b2)
        recall_doc2vec.append(c2)
        f1_doc2vec.append(d2)
        

        kfold_acc.append([acc_ifidf,acc_pca_ifidf,acc_glove,acc_doc2vec])
        kfold_prec.append([prec_ifidf,prec_pca_ifidf,prec_glove,prec_doc2vec])
        kfold_recall.append([recall_ifidf,recall_pca_ifidf,recall_glove,recall_doc2vec])
        kfold_f1.append([f1_ifidf,f1_pca_ifidf,f1_glove,f1_doc2vec])

        '''
        pca = PCA(n_components = 3)
        # evaluation with graph
        #ifidf_data = np.concatenate( (ifidf_train_vectors.toarray(), ifidf_test_vectors.toarray()), axis =0 )
        ifidf_data =  ifidf_test_vectors
        pca_ifidf_labels =  ifidf_train_labels + ifidf_test_labels
        pca_ifidf_labels = ifidf_test_labels
        
        #glove_data = np.concatenate((glove_train_vectors,glove_test_vectors), axis = 0)
        glove_data = glove_test_vectors
        #pca_glove_labels =  glove_train_labels + glove_test_labels
        pca_glove_labels = glove_test_labels
        
        
        #doc2vec_data = np.concatenate((doc2vec_train_vectors,doc2vec_test_vectors), axis = 0)
        doc2vec_data = doc2vec_test_vectors
        #pca_doc2vec_labels =  doc2vec_train_labels + doc2vec_test_labels
        pca_doc2vec_labels = doc2vec_test_labels
        
        #getting principle component
        pca_ifidf = pca.fit_transform(ifidf_data.toarray())
        
        eval_graph.plot(pca_ifidf, pca_ifidf_labels, should_ignore_brand,'scatter-ifidf')
        
        
        pca_glove = pca.fit_transform(glove_data)
        eval_graph.plot(pca_glove, pca_glove_labels, should_ignore_brand,'scatter-glove')

        pca_doc2vec = pca.fit_transform(doc2vec_data)
        eval_graph.plot(pca_doc2vec, pca_doc2vec_labels, should_ignore_brand,'scatter-doc2vec')
        '''
        print 'DONE'
        print kfold_acc
        print kfold_prec
        print kfold_recall
        print kfold_f1

        results.write('\n%s fold \n' % k)
        results.write('Accuracy: \n')
        for item in kfold_acc[k]:
            results.write("%s " % item)
            
        results.write('\nPrecision: \n')
        for item in kfold_prec[k]:
            results.write("%s  " % item)

        results.write('\nRecall: \n')
        for item in kfold_recall[k]:
            results.write("%s  " % item)

        results.write('\nF1: \n')
        for item in kfold_f1[k]:
            results.write("%s  " % item)

    results.close()
    print 'Accuracy'
    print kfold_acc
    
    print 'Precision'
    print kfold_prec

    print 'Recall'
    print kfold_recall
    
    print 'f1-score'
    print kfold_f1
    
   
    
    
if __name__=="__main__":
    main()
