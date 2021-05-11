'''Script to convert any MVTec dataset to a dataset compatible 
    with all the other scripts:
    
    How to use: 
        download a MVTec dataset from here:
        https://www.mvtec.com/company/research/datasets/mvtec-ad/
        
        and unzip it in the data folder,
        
        then simply run mvtec_dataset.sh script which will trigger this script
        after some pre-processing
'''

import pandas as pd 
import csv
import os
import re
import sys

dataset = sys.argv[1]

with open(f'data/{dataset}/{dataset}.csv', 'w') as csvfile:
    fieldnames = ['image_name', 'label', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for filename in os.listdir(f'data/{dataset}/img'):
        re_search = re.search(r"(.+)_(.+)_(\d+).png", filename)
        datatype, label, img_name = re_search.group(1), re_search.group(2), re_search.group(3)
        label = 0 if label == 'good' else 1
        writer.writerow({'image_name': filename, 'label': label, 'type': datatype})
