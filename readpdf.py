import pandas as pd

import sys
from pprint import pprint

FILE = "C:/Users/DELL/Downloads/test.pdf"

def readfromfile_camelot(file):
    import camelot
    try:
        tables = camelot.read_pdf(file, pages='all')
        print("Total tables extracted:", tables.n)
        df = tables[0].df
        #print(df)
        #df.to_csv("test_camelot.csv")
    except:
        print(f"{file} loading failed due to {sys.exc_info()[0]}")
        return "", 
    print("Table 1: Contents:")
    for index, row in df.iterrows(): 
        print(index+1, end=": ")
        [print(((row[i].replace('\n','')).strip()),end="; ") for i in range(7)]  #TODO: Get the number of columns, .encode('utf-8')
        print()
    return tables[0].df

def readfromfile_tika(file):
    try:
        from tika import parser
        file_data = parser.from_file(file)
        file_data = file_data['content'].encode('utf-8', errors='ignore') 
        file_data = file_data.decode('utf-8') 
        print(file_data)
    except:
        print(file, "loading failed")
        return "", 
    return file_data

def readfromfile_tabula(file):
    try:
        from tabula import read_pdf, convert_into
        file_data = read_pdf(file, output_format = "json", pages='all')
        #convert_into(FILE, "test.csv", output_format="csv", pages='all'
    except:
        print(file, "loading failed")
        return "", 
    return file_data

option = sys.argv[1]
if option =="tika":
    file_data = readfromfile_tika(FILE)
elif option == "tabula":
    df = readfromfile_tabula(FILE)
    data = df[0]['data']
    nrows = len(data)
    for nrow in range(nrows):
        row = data[nrow]
        ncols = len(row)
        print(f"{nrow}:", end=" ")
        for ncol in range(ncols):
            print(f"{ncol}: {row[ncol]}",end=" ")
            #content = row[ncol]['text'].encode('utf-8', errors='ignore')
            #print(f"{ncol}: {content.decode('utf-8')}",end=" ") #['text'].encode('utf-8', errors='ignore') 
        print()
elif option == "camelot":
    readfromfile_camelot(FILE)
#pprint(df)