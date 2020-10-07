import pandas as pd 
import csv
import os


with open('data/carpet/carpet.csv', 'w') as csvfile:
    fieldnames = ['image_name', 'label', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for filename in os.listdir('data/carpet/img'):
        datatype, label, img_name = filename.split('_')
        label = 0 if label == 'good' else 1
        writer.writerow({'image_name': filename, 'label': label, 'type': datatype})