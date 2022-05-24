import categorize_nn
from pathlib import Path
import sys, os, datetime

# python test.py E:\Sree\netalytics\SensitiveDataDiscovery\data\sdd1\train\bank categorize
# python test.py E:\Sree\netalytics\SensitiveDataDiscovery\data\sdd3\train\bank learn bank E:\Sree\netalytics\SensitiveDataDiscovery\data\online\validate

def getf(path, option, category=""):
    print("getf()", path, option, category)
    global nfiles, ndirectories, nlevel
    files = os.listdir(path)
    folder = Path(path)
    for file in files:
        #print("getf()", file)
        f = str(folder / file)
        if os.path.isdir(f):
            print(f, "is a directory")
            '''print("\t"*nlevel, f)
            ndirectories += 1 
            nlevel += 1'''
            getf(f, option)
        else:
            #nfiles += 1
            #print("\t"*nlevel,file)
            if option == 'scan':
                #text.getcorpus(path)
                text.readfromfile(f)
            elif option == 'categorize':
                print("Processing started for ", f)
                category = categorize_nn.predict(f)
                print(f, " --> ", category)
                resultfile.write(f + "," + category + "\n")
            elif option == 'learn':
                learn_nn_online.learn(folder, file, category, validate)

path = sys.argv[1]
option = sys.argv[2]
category, validate = "", ""
if option == 'categorize':
    tic = datetime.datetime.now()
    resultfile = open("out/out" + tic.strftime("_%Y%m%d_%H%M")+ ".csv", "a")
    resultfile.write("file, category\n")
elif option == 'scan':
    import text
elif option == 'learn':
    import learn_nn_online
    category = sys.argv[3]
    validate = sys.argv[4]
getf(path, option, category)
if option == 'categorize':
    resultfile.close()