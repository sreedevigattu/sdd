'''
1. datapreparation
    a. readfromfile: text.py
    b. tokenize: tokenize
    c. preparedata: one-hot encoding + pad sequences >> max_length
2. predict

todo: if training, pickle max+length else read from pkl file
'''

from keras.models import load_model
import numpy as np
import joblib
import sys, os, datetime
import text, utility

def predict(f):
    corpusd_p = {}
    meta, data = text.readfromfile(f)
    category_maj = "none"
    if data == "":
        return category_maj
    corpusd_p[f] = data
    corpus_p = utility.tokenize(corpusd_p)
    corpus_p, vocab_size, max_length = utility.preparedata(corpus_p)
    print("Test corpus:", type(corpus_p), corpus_p.ndim, corpus_p.shape)
    prediction = model.predict(corpus_p)
    print("Predicted:", prediction)
    
    for doc, category in zip(corpusd_p.keys(), prediction):
        c =  np.argmax(category, axis=0)
        msg = 'none'
        if category[c] > 0.7:
            msg = 'ok'
            category_maj = categories_s[int(c)]
        print('%r => %s, %f %s %s %s' % (doc, category,  category[c], c, category_maj, msg ))
    return category_maj

from keras.models import load_model
model = load_model(os.getcwd() + "/class/sddmodel.h5")
print("/class/sddmodel.h5")
categories_s = joblib.load(os.getcwd() + "/class/sddcategories.pkl") # Using pickled corpus