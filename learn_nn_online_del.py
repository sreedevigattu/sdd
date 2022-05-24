import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pprint import pprint
import sys, os, random, datetime
from time import time
import process, text, utility

# python sdd/learn_nn_online.py E:/Sree/netalytics/SensitiveDataDiscovery/data/sdd1/ text 3 1 > out/out.txt

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def CORPUS(beforename):
    print(os.getcwd() + "/corpus/" + prefix + "_" + token + "_" + beforename)
    return os.getcwd() + "/corpus/" + prefix + "_" + token + "_" + beforename

def getcorpus(corpusd_t, categoriesd_t, corpusd_v, categoriesd_v): 
    tic = datetime.datetime.now()
    if (os.path.exists(CORPUS("train_token_corpus.pkl")) and 
            os.path.exists(CORPUS("validate_token_corpus.pkl"))):
        corpus_t = joblib.load( CORPUS("train_token_corpus.pkl")) # Using pickled corpus
        corpus_v = joblib.load( CORPUS("validate_token_corpus.pkl")) # Using pickled corpus
        toc = datetime.datetime.now()
        print("Tokenized from pickle files", toc-tic)
    else:
        #corpus_t = list(corpusd_t.values());     corpus_v = list(corpusd_v.values())
        # convert each token to a NER type + shape format
        corpus_t = utility.tokenize(corpusd_t, categoriesd_t)
        corpus_v = utility.tokenize(corpusd_v, categoriesd_v)
        #print("corpus_t:", corpus_t[0])
        #print("corpus_v:", corpus_v[0])
        joblib.dump(corpus_t, CORPUS("train_token_corpus.pkl") , compress=9)
        joblib.dump(corpus_v, CORPUS("validate_token_corpus.pkl") , compress=9)
        toc = datetime.datetime.now()
        print("Tokenized from scratch", toc-tic)
    return corpus_t, corpus_v #, corpus_p

def convert2int(categories):
    categories = [categories_n[s] for s in categories]
    return categories

print("Started")
startDT = process.recordstarttime(sys.argv)
dataset_folder = sys.argv[1]
token = sys.argv[2]
NUM_CLASSES = int(sys.argv[3])
epochs =  int(sys.argv[4])
temp = dataset_folder.split('/')
prefix = temp[len(temp)-2]
 
# 1. Read the data from the files
if (os.path.exists(CORPUS("train_corpus.pkl")) and os.path.exists(CORPUS("train_categories.pkl")) and 
    os.path.exists(CORPUS("validate_corpus.pkl")) and os.path.exists(CORPUS("validate_categories.pkl"))): 
    tic = datetime.datetime.now()
    corpusd_t = joblib.load(CORPUS("train_corpus.pkl")) # Using pickled corpus
    categoriesd_t = joblib.load(CORPUS("train_categories.pkl")) # Using pickled corpus
    corpusd_v = joblib.load( CORPUS("validate_corpus.pkl")) # Using pickled corpus
    categoriesd_v = joblib.load( CORPUS("validate_categories.pkl")) # Using pickled corpus
    toc = datetime.datetime.now()
    print("Loaded corpus from pickled files", toc-tic)
else:
    tic = datetime.datetime.now()
    TRAIN_DIR = dataset_folder +'train/'
    VALIDATE_DIR = dataset_folder +'validate/'
    corpusd_t, categoriesd_t = text.getcorpusx(TRAIN_DIR)
    corpusd_v, categoriesd_v = text.getcorpusx(VALIDATE_DIR)
    toc = datetime.datetime.now()
    print("Loaded corpus from data files", toc-tic)
    pprint(categoriesd_t)
    pprint(categoriesd_v)

    joblib.dump(corpusd_t, CORPUS("train_corpus.pkl") , compress=9)
    joblib.dump(categoriesd_t, CORPUS("train_categories.pkl") , compress=9)
    joblib.dump(corpusd_v, CORPUS("validate_corpus.pkl") , compress=9)
    joblib.dump(categoriesd_v, CORPUS("validate_categories.pkl") , compress=9)

def compute_regret(model,X,Y):
    predicted_classes = model.predict_classes(X, verbose=0)
    correct_indices = np.nonzero(predicted_classes == Y)[0]
    incorrect_indices = np.nonzero(predicted_classes != Y)[0]
    return (correct_indices, incorrect_indices)

def vis(figno, n, no_of_samples, train_list, test_list):
    plt.figure(figno)
    X_vals = [i for i in range(0,no_of_samples,n)]

    Train_accuracy = [val[1] for val in train_list]
    Test_accuracy = [val[1] for val in test_list]

    print("no_of_samples=",no_of_samples, "X_vals=",len(X_vals),"train_list=", len(train_list))

    axes = plt.gca()
    x_min = X_vals[0]
    x_max = X_vals[-1]+1
    axes.set_xlim([x_min,x_max])

    plt.scatter(X_vals, Train_accuracy, color='r')
    plt.plot(X_vals, Train_accuracy, color='r', label='# Training accuracy')

    plt.scatter(X_vals, Test_accuracy, color='g')
    plt.plot(X_vals, Test_accuracy, color='g', label='# Testing accuracy')

    plt.xlabel('# of Data Points')
    plt.ylabel('# of Regrets')
    plt.title('Training & Test Data -  # of Regrets vs # of Data Points')
    plt.legend()

    plt.show()

size_t = len(corpusd_t); size_v = len(corpusd_v)
corpus_t, corpus_v = getcorpus(corpusd_t, categoriesd_t, corpusd_v, categoriesd_v)
categories_t = categoriesd_t.values(); categories_v = categoriesd_v.values()

categories = list(enumerate(list(set(categories_t))))
categories_n, categories_s = {}, {}
for n,s in categories:
    print(n,s)
    categories_n[s] = n
    categories_s[n] = s
joblib.dump(categories_s,os.getcwd() + "/class/sddcategories.pkl")
categories_t = convert2int(categories_t); categories_v = convert2int(categories_v)
print(categories_t)
print(categories_v)

print("Training data:", type(corpus_t), len(corpus_t),"Validation data:", type(corpus_v), len(corpus_v))
print("Training categories:", set(categories_t), "Validation categories", set(categories_v))

# 2a. Prepare data for Embedding Layer
corpus_t, vocab_size, max_length = utility.preparedata(corpus_t, True)
corpus_v, vocab_size, max_length = utility.preparedata(corpus_v)
print("Training corpus:", type(corpus_t), corpus_t.ndim, corpus_t.shape)
print("Validation corpus:", type(corpus_v), corpus_v.ndim, corpus_v.shape)

# For multi-class - start
from keras.utils import to_categorical
categories_t = to_categorical(categories_t)
categories_v = to_categorical(categories_v)
print("Training categories:", type(categories_t), categories_t.ndim, categories_t.shape)
print("Validation categories:", type(categories_v),categories_v.ndim, categories_v.shape)
# For multi-class - end
print(flush=True)

# 3 Define the model
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

embedding_dim = 100 #50
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

# For multi-class - start
#model.add(layers.Flatten())
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# For multi-class - end
# For 2-class - start
'''model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])'''
# For 2-class - end
print(model.summary())

train_regret_list, test_regret_list  = [], []
train_score_list, test_score_list = [], []
REGRET_CHECKPOINT, ITERATION_CHECKPOINT = 2, 2
start_time, end_time = 0, 0
categories_t = np.asarray(categories_t)
no_of_samples = len(corpus_t)
for i in range(no_of_samples):
    if ((i+1)%ITERATION_CHECKPOINT == 0):
        print("Example : {}".format(i))
        end_time = time()
        time_lapse = end_time - start_time
        print("Training on {} point took {} secs".format(ITERATION_CHECKPOINT, (end_time - start_time)))
        start_time = time()

    history = model.fit(corpus_t[i:i+1], categories_t[i:i+1], epochs=epochs, verbose=False,
                        validation_data=(corpus_v, np.asarray(categories_v)),batch_size=3)

    if ((i+1)%REGRET_CHECKPOINT == 0):
        # compute regrets and store it
        train_regret = compute_regret(model,corpus_t, categories_t)
        test_regret = compute_regret(model, corpus_v, categories_v)
        train_regret_list.append(train_regret)
        test_regret_list.append(test_regret)

        train_score = model.evaluate(corpus_t, categories_t, verbose=0)
        test_score = model.evaluate(corpus_v, categories_v, verbose=0)
        train_score_list.append(train_score)
        test_score_list.append(test_score)

print("train_regret_list=",len(train_regret_list),"test_regret_list=",len(test_regret_list))
print("train_score_list=",len(train_score_list),"test_score_list=",len(test_score_list))

vis(3, REGRET_CHECKPOINT, no_of_samples, train_regret_list, test_regret_list)
vis(4, REGRET_CHECKPOINT, no_of_samples, train_score_list, test_score_list)

correct, incorrect = compute_regret(model,corpus_t,categories_t)
print(len(correct), len(incorrect))
print(train_score_list[-1][1], test_score_list[-1][1])

from keras.models import load_model
model.save(os.getcwd() + '/class/sddmodel_online.h5')  # creates a HDF5 file
del model  # deletes the existing model
process.recordstoptime(startDT)