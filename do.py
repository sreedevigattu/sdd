import text
import categorize
from pathlib import Path
import queue, threading
import sys, os
import socket

# 1. FileMonitor watches the configured list of folders for changes. 
# 2. Once a change is detected it adds it to the procsesing queue. 
# 3. Multiple worker threads read from the queue and categorise the file content 
#    and store the result (file, location, category) in es-category index

from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': '45.77.46.58', 'port': 9200}])
category_index = "categories"
hostname = socket.gethostname()

# TODO: Configuration file - # No. threads to categorize and add to elastic search

def do_work(tname, file):
    meta, data = text.readfromfile(file)
    content_l = []
    content_l.append(data)
    category = categorize.categorize(file,content_l).upper()
    out = "[" + tname + "]" + file + ":" + category
    print(out)
    f = file.split('/')
    filename = f[len(f)-1]
    filepath = ""
    for i in range(len(f)-1):
        filepath += f[i] + '/'
    try:
        entry={
                "host": hostname,
                "location": filepath,
                "filename": filename,
                "category": category,
        }
        es.index(index=category_index, doc_type='files', id=file, body=entry)
    except Exception as ex:
        print('Error in indexing data')
        print(str(ex))

def add_to_queue(dir):
    files = os.listdir(dir)
    for file in files:
        f = dir + '/' + file
        q.put(f)

# Read an item from the queue and peform the task
def worker():
    while True:
        f = q.get()
        if f is None:
            break
        do_work(threading.current_thread().name, f)
        q.task_done()

# Create queue and threads
print("Create queue and threads")
num_worker_threads = 2
q = queue.Queue()
threads = []
for i in range(num_worker_threads):
    t = threading.Thread(target=worker, name='T'+str(i+1))
    t.start()
    threads.append(t)

# Read the folders from a configuration file
paths = []
f = open(os.getcwd() + "/config/config", "r")
for line in f:
        print(line.rstrip())
        paths.append(line.rstrip())
f.close()

from fsmonitor import FSMonitor
fsm = FSMonitor()

# Add items to the queue
for path_ in paths:
        add_to_queue(path_)      
        watch = fsm.add_dir_watch(path_)

while True:
        # TODO: Does this work if many files are created together
        # TODO: What if the fsmonitor crashes, all notifications will be lost? 
        for evt in fsm.read_events():
                print("%s %s %s" % (evt.action_name, evt.watch.path, evt.name))
                if evt.action_name == 'create': #in [ 'create', 'modify']:
                        q.put(evt.watch.path + "/" +  evt.name)
                        # TODO: modify: Update entry in elasticsearch
                elif evt.action_name == 'delete':
                        print("To delete entry from elasticsearch")
                        es.delete(index=category_index,doc_type='files',id=evt.watch.path + "/" +  evt.name)

# block until all tasks are done
q.join()

# stop workers
for i in range(num_worker_threads):
    q.put(None)
for t in threads:
    t.join()

'''
# Test - Begin
import time, random
def do_work(tname, item):
t = random.randint(1,6)
out = tname + "-" + item + "-" + str(t) + " "
print(out)
time.sleep(t)

def add_to_queue():
for item in ["Mon","Tues","Wed","Thurs","Fri","Sat","Sun"]:
        q.put(item)
# Test - End
'''