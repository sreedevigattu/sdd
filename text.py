from tika import parser
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from pathlib import Path
import sys, os, string, re, datetime
from pprint import pprint

import tika
from tika import detector

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
'''
classifier:
    pklprefix + 'class_'+algorithm+'tfidf.pkl'
    pklprefix + 'class_'+algorithm+'.pkl'
'''
def getclassifier(dataset_folder,algorithm,transformer):
    return getpklprefix(dataset_folder) + 'class_' + algorithm + transformer + '.pkl'

def getprefix(dataset_folder,folder):       # data/pkl/_train_<file>
    return dataset_folder + "/pkl/" + '_' + folder + '_'

def getpklprefix(dataset_folder):    # pkl/data_<file>
    return dataset_folder + "/pkl/"

import en_core_web_md
nlp = en_core_web_md.load()
nlp.max_length = 2000000 # default = 1000000
ETYPE2COUNT = {}
ETYPE2WORDS = {}
def usespacy(sentence):
    doc = nlp(sentence)
    #print(len(doc.ents))
    #pprint([(ent.text, ent.label_) for ent in doc.ents])
    #pprint([(word, word.ent_iob_, word.ent_type_) for word in doc ]) #if word.ent_type_
    words = []
    for word in doc:
        #pprint((word, word.ent_iob_, word.ent_type_, word.pos_, word.tag_, word.dep_, word.shape_))
        #global NORP
        if word.ent_type_ != '':
            '''if word.ent_type_ == 'NORP':
                NORP += str(word) + "-" + word.shape_ + "\n"'''
            key = word.ent_type_ + '_'+ word.shape_
        else:
            key = 'UNK_'+ word.pos_ + '_' + word.shape_
        words.append(key)
        if key in ETYPE2COUNT.keys():
            ETYPE2COUNT[key] += 1
            ETYPE2WORDS[key].append(word)
        else:
            ETYPE2COUNT[key] = 1
            ETYPE2WORDS[key] = [ word ]
    return words

def my_tokenizer(doc):
    doc = doc.lower()
    tokens = nltk.word_tokenize(doc)
    return tokens
COUNT_ = 0
def typetokenizer(doc):
    global COUNT_
    COUNT_ += 1
    '''if COUNT_ % 100 == 0:
        print(COUNT_, "typetokenize", flush = True)
    else:
        print(COUNT_, "typetokenize")'''
    #print(type(doc), len(doc))
    #doc = doc.decode('utf-8')
    doc = doc.lower()
    #tokens = usespacy(doc)
    MAX = 100000    
    doclen = len(doc)
    n = (doclen//MAX) + 1
    tokens = []
    for i in range(n):
        START = i * MAX
        if doclen > (i+1)*( MAX-1):
            END = (i+1)*( MAX-1)
        else:
            END = doclen
        doc1 = doc[START:END]
        tokens1 = usespacy(doc1)
        tokens.extend(tokens1)
    return tokens

def removeblanks(text):
    text = re.sub('\n+',' ',text)
    text = re.sub('\t+',' ',text)
    text = re.sub(' +',' ',text)
    return text

def updatefeatures(data, authors, creationdates):
    if 'Author' in data:
        authors.append(data['Author'])
    else:
        authors.append('')
    if 'Creation-Date' in data:
        creationdates.append(data['Creation-Date'])
    else:
        creationdates.append('')
    #print(data['resourceName'],data['Content-Type'])

# TODO: Optimize: use list comprehensions and lambdas
def readfromfile(file):
    try:
        file_data = parser.from_file(file)
    except:
        print(file, "loading failed")
        return "", ""
    meta = ""; safe_text = ""
    if 'metadata' in file_data.keys():
        meta = file_data['metadata']
    if 'content' in file_data.keys():
        text = file_data['content']
        if text is None:
            print(file, "does not contain text")
            #return meta, ""
            text = ""
        #else:
        text = removeblanks(text)
        safe_text = text.encode('utf-8', errors='ignore') # TODO: Why utf-8 is needed? If utf-8 is not given, .txt files do not work
    if meta == "" and safe_text =="":
        print(file,file_data, "ERROR - empty meta and content")
        safe_text = safe_text.encode('utf-8', errors='ignore')
    #print(file, '--->', detector.from_file(file), '-->', detector.from_buffer(safe_text)) # MIME 
    #pprint(data)
    print(file, end=" ")
    if meta != "" and 'Content-Type' in meta:
        print(meta['Content-Type'], end=" ")
    print(len(safe_text), type(safe_text))
    return meta, safe_text

def readfromfileex(folder, file,category, corpusd, categoriesd, metad, size):
    f = str(folder / file)
    if os.path.isdir(f):
        print(f,"is a directory") 
        return
    meta, data = readfromfile(f)
    '''print(f, end=" ")
    if meta != "" and 'Content-Type' in meta:
        print(meta['Content-Type'], end=" ")
    print(len(data), type(data))'''
    size.append(len(data))
    corpusd[f] = data
    categoriesd[f] = category
    metad[f] = meta
    if len(corpusd) % 10 == 0:
        print(flush=True)
 
def _getcorpus(path,category):
    print("Parsing files from",path)
    corpusd, categoriesd, metad = {}, {}, {}
    size, authors, creationdates = [],[],[]
    files = os.listdir(path)
    folder = Path(path)
    [readfromfileex(folder, file,category, corpusd, categoriesd, metad, size) for file in files]
    [updatefeatures(data, authors, creationdates) for data in list(metad.values())]  
    return corpusd,categoriesd, size, authors, creationdates

def getcorpus(path):
    return _getcorpus(path,"")

def getcorpusx(path):
    print("Parsing files from",path)
    corpusd, categoriesd = {}, {}
    size, authors, creationdates = [],[],[]
    files = os.listdir(path)
    folder = Path(path)
    for file in files:
        f = str(folder / file)
        if os.path.isdir(f):
            print(f,"is a directory") 
            category = file
            corpusd_t, categoriesd_t, size_t, authors_t, creationdates_t = _getcorpus(f+'/',category)
            corpusd.update(corpusd_t); categoriesd.update(categoriesd_t)
            size.extend(size_t); authors.extend(authors_t); creationdates.extend(creationdates_t)
    return corpusd, categoriesd

folder = sys.argv[1] 
corpusd, categoriesd, size, authors, creationdates = getcorpus(folder)
for name, doc in corpusd.items():
    tokens = typetokenizer(doc.decode('utf-8'))
    print(name, len(doc), type(doc), type(tokens), len(tokens))
print(list(set(tokens)))
pprint(ETYPE2WORDS)
pprint(ETYPE2COUNT)