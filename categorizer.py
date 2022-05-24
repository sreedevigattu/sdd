from pathlib import Path
import text
import categorize
import sys, os

# files in a folder --> categorize --> es.category
# For each content in the elasticsearch doc index, find the category and 
# write into the elasticsearch category index

outliers = {'The President did not comment':-1,
			'I lost the keys':-1,
			'The team won the game':-1,
			'Sara has two kids':-1,
			'The ball went off the court':-1,
			'They had the ball for the whole game':-1,
			'The show is over':-1
			}	

for doc in outliers.keys():
    print(doc, "Category:", categorize.categorize(doc,doc).upper())

from elasticsearch import Elasticsearch
from datetime import datetime
date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
print("Hi there",date_time)	
es = Elasticsearch([{'host': '45.77.46.58', 'port': 9200}])
	
path_ = sys.argv[1]
files = os.listdir(path_)
folder = Path(path_)
for file in files:
    f = str(folder / file)
    meta, data = text.readfromfile(f)
    content_l = []
    content_l.append(data)
    category = categorize.categorize(f,content_l).upper()
    print(f, "Category:", category )
    try:
    	entry={
    			"filename":f,
				"category": category,
    			"location":doc['_source']['path']['real'],
		}
    	res = es.index(index=category_index, doc_type='day', body=entry)
    except Exception as ex:
		print('Error in indexing data')
		print(str(ex))