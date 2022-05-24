import nltk
import pandas as pd
import text
import os, datetime

# todo: For structured files, don't have to convert it to dataframe.Just read each line is sufficient
def process_csv(dataset_folder, tokenizer, w2v):
    print("process_csv")
    count, start, end = 0, 0, 20
    sentences, categories = [] , []
    files = os.listdir(dataset_folder)
    for FILE in files:
        f = dataset_folder+FILE
        extn = f.split('.')[1]
        print(f,extn)
        if extn == "csv":
            df = pd.read_csv(f)
        elif extn == "xlsx":
            df = pd.read_excel(f)
        else:
            print("Unsupported file type", extn)
            continue

        for item in df.itertuples(index=False):
            if count < start:
                count += 1
                continue
            '''if count > end:
                break'''
            sentence = ""
            for name, value in item._asdict().items():
                sentence += str(value) + " "   
            if w2v == True:
                sentence = tokenizer(sentence)          
            sentences.append(sentence)
            categories.append(1)
            count += 1
    return sentences, categories

def recordstarttime(argv):
    startDT = datetime.datetime.now()
    print(20*"====")
    print("Started at", startDT, end = "-->")
    [print(arg, end=" ") for arg in argv]; print()
    print(20*"----")
    return startDT

def recordstoptime(startDT):
    print(20*"----")
    stopDT = datetime.datetime.now()
    print("Completed at", stopDT, "Processing Time:", stopDT - startDT)
    print(20*"====")
    return stopDT