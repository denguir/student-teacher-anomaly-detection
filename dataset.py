'''Script to convert any MVTec dataset to a dataset compatible 
   with all the other scripts.
   This script is triggered by dataset.sh'''
    
import pandas as pd 
import csv
import os
import sys

dataset = sys.argv[1]

with open(f'data/{dataset}/{dataset}.csv', 'w') as csvfile:
    fieldnames = ['image_name', 'label', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for filename in os.listdir(f'data/{dataset}/img'):
        datatype, label, img_name = filename.split('_')
        label = 0 if label == 'good' else 1
        writer.writerow({'image_name': filename, 'label': label, 'type': datatype})
