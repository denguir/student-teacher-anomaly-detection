'''
Script to init a CSV file from MVTec dataset.
'''


import csv
import os
import sys

dataset = sys.argv[1]
    

with open(f'data/{dataset}/{dataset}.csv', 'w') as csvfile:
    fieldnames = ['image_name', 'label', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(f'data/{dataset}/img'):
        fname = filename.split('_')
        datatype = fname[0]
        label = fname[1]
        label = 0 if label == 'good' else 1
        writer.writerow({'image_name': filename, 'label': label, 'type': datatype})
