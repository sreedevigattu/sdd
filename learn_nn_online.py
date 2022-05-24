import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pprint import pprint
import sys, os, random, datetime
import process, text, utility
from keras.models import Model, load_model

# python learn_nn_online.py E:/Sree/netalytics/SensitiveDataDiscovery/data/sdd3/train/bank
def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1); plt.title('Training and validation accuracy')
    plt.plot(history.history['acc']); plt.plot(history.history['val_acc']);  
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(['Train','Validation'])

    plt.subplot(1, 2, 2); plt.title('Training and validation loss')
    plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(['Train','Validation'])

def CORPUS(beforename):
    print(os.getcwd() + "/corpus/" + prefix + "_" + token + "_" + beforename)
    return os.getcwd() + "/corpus/" + prefix + "_" + token + "_" + beforename

def getcorpus(corpusd_t, categoriesd_t, corpusd_v, categoriesd_v): 
    #corpus_t = list(corpusd_t.values());     corpus_v = list(corpusd_v.values())
    # convert each token to a NER type + shape format
    corpus_t = utility.tokenize(corpusd_t, categoriesd_t, token)
    corpus_v = utility.tokenize(corpusd_v, categoriesd_v, token)
    #print("corpus_t:", corpus_t[0])
    #print("corpus_v:", corpus_v[0])   
    return corpus_t, corpus_v #, corpus_p

def convert2int(categories):
    categories = [categories_n[s] for s in categories]
    return categories

categories_n = {}; token = "text"; epochs = 20
print("Loading the model")
if os.path.exists(os.getcwd() + "/class/sddmodel_ol.h5") == True:
    model = load_model(os.getcwd() + "/class/sddmodel_ol.h5")
    print("--> /class/sddmodel_ol.h5")
else:
    model = load_model(os.getcwd() + "/class/sddmodel.h5")
    print("/class/sddmodel.h5")
model.summary()
model_json = model.to_json()
with open("out\model.json", "w") as json_file:
    json_file.write(model_json)
from keras.utils import plot_model
plot_model(model, show_shapes = True, to_file='out\model.png')
categories_s = joblib.load(os.getcwd() + "/class/sddcategories.pkl")
CLASSES = len(categories_s)

def learn(folder, file, category, validate_dataset_folder):  
    print("learn",folder, file, category)
    global model
    model.summary()
    print("configuration", model.get_config())

    print(CLASSES,token, epochs); print(categories_n); print(categories_s)

    VALIDATE_DIR = validate_dataset_folder
    corpusd_v, categoriesd_v = text.getcorpusx(VALIDATE_DIR)
    pprint(categoriesd_v)

    corpusd_t, categoriesd_t, metad = {}, {}, {}
    size, authors, creationdates = [],[],[]
    text.readfromfileex(folder, file, category, corpusd_t, categoriesd_t, metad, size)         
    pprint(categoriesd_t)

    size_t = len(corpusd_t); size_v = len(corpusd_v)
    corpus_t, corpus_v = getcorpus(corpusd_t, categoriesd_t, corpusd_v, categoriesd_v)
    categories_t = categoriesd_t.values(); categories_v = categoriesd_v.values()

    for n,s in categories_s.items():
        print(n,s)  
        categories_n[s] = n
    categories_t = convert2int(categories_t); categories_v = convert2int(categories_v)
    print(categories_t); print(categories_v)

    print("Training data:", type(corpus_t), len(corpus_t),"Validation data:", type(corpus_v), len(corpus_v))
    print("Training categories:", set(categories_t), "Validation categories", set(categories_v))

    # 2a. Prepare data for Embedding Layer
    corpus_t, vocab_size, max_length = utility.preparedata(corpus_t)
    corpus_v, vocab_size, max_length = utility.preparedata(corpus_v)
    print("Training corpus:", type(corpus_t), corpus_t.ndim, corpus_t.shape)
    print("Validation corpus:", type(corpus_v), corpus_v.ndim, corpus_v.shape)

    from keras.utils import to_categorical
    categories_t = to_categorical(categories_t,num_classes=CLASSES)
    categories_v = to_categorical(categories_v,num_classes=CLASSES)
    print(categories_t)
    print(categories_v)
    print("Training categories:", type(categories_t), categories_t.ndim, categories_t.shape)
    print("Validation categories:", type(categories_v),categories_v.ndim, categories_v.shape)
    print(flush=True)

    # 3 Load the model
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

    # Save the complete model - architecture, weight values (which were learned during training)
    # training config (what you passed to compile), the optimizer and its state, if any 
    # (this enables you to restart training where you left off)
    model.save(os.getcwd() + '/class/sddmodel_ol.h5') 
    print("saved model", '/class/sddmodel_ol.h5', datetime.datetime.now())


