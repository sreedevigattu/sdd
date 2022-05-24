from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import os, joblib
import text

def tokenize(corpusd, categoriesd={}, token = ""):
    #print("token=",token)
    temp = []
    for fname, corpus in corpusd.items():
        tokentype = "text"
        if fname in categoriesd.keys(): 
            if categoriesd[fname] in ['bank']:
                tokentype = token
        #print(fname, categoriesd[fname], tokentype, type(corpus))
        if tokentype == "text":
            tokenizer = text.my_tokenizer
        elif tokentype == "value":
            tokenizer = text.typetokenizer
        corpus = tokenizer(corpus.decode('utf-8'))
        temp.append(corpus)
    return temp

def preparedata(corpus, trainingdataset = False): #categoriesd={},
    #corpus = tokenize(corpusd, categoriesd)
    vocab_size = 1000 # todo: read from config file
    separator = ' ' 
    corpus_ = [separator.join(d) for d in corpus]
    encoded_docs = [one_hot(d, vocab_size) for d in corpus_]
    #print("Original:", corpus_[0][0:100])
    print(len(corpus_), len(encoded_docs))
    print("Encoded:", encoded_docs[0][0:10])

    if trainingdataset == True:
        doc_lens = [ len(d) for d in encoded_docs]
        print("Encoded docs:", len(encoded_docs), doc_lens, max(doc_lens))
        max_length = max(doc_lens)
        joblib.dump(max_length, os.getcwd() + "/class/sddconfig.pkl" , compress=9)
    else:
        max_length = joblib.load(os.getcwd() + "/class/sddconfig.pkl") 
    print(type(max_length), max_length)   

    # pad documents to a max length
    corpus = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print("Padded docs:", len(corpus))
    return corpus, vocab_size, max_length