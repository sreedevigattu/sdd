import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pprint import pprint
import sys, os, random, datetime
import process, text, utility

# python learn_nn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/sdd1/ text 3 20 noshow> out/sdd1_text_3_e20.txt
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
        corpus_t = utility.tokenize(corpusd_t, categoriesd_t, token)
        corpus_v = utility.tokenize(corpusd_v, categoriesd_v, token)
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
from keras import optimizers

embedding_dim = 100 #50
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #Adam
'''sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])'''
print(model.summary())

history = model.fit(corpus_t, np.asarray(categories_t), epochs=epochs, verbose=False,
                    validation_data=(corpus_v, np.asarray(categories_v)),batch_size=3)
loss, accuracy = model.evaluate(corpus_t, categories_t,batch_size=3, verbose=False)
print("Loss:", loss, "Training Accuracy: {:.4f}".format(accuracy))
validation = model.predict(corpus_v)
print("Actual:", categories_v)
print("Predicted:", validation)
for doc, category in zip(corpusd_v.keys(), validation):
    c =  np.argmax(category, axis=0)
    msg = 'none'
    if category[c] > 0.7:
        msg = 'ok'
    print('%r => %s, %f %s %s %s' % (doc, category,  category[c], c, categories_s[int(c)], msg ))
loss, accuracy = model.evaluate(corpus_v, categories_v, verbose=False)
print("Loss:", loss,"Testing Accuracy:  {:.4f}".format(accuracy))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(categories_v.argmax(axis=1), validation.argmax(axis=1)))
plot_history(history)
plt.show()

from keras.models import load_model
model.save(os.getcwd() + '/class/sddmodel.h5')  # creates a HDF5 file
del model  # deletes the existing model
process.recordstoptime(startDT)