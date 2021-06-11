#!/bin/bash

# HOW TO USE:
# download a MVTec dataset from here:
# https://www.mvtec.com/company/research/datasets/mvtec-ad/
# and unzip under the data folder
# and run this script

dataset="$1"
img_dir="data/$dataset/img"
test_dir="data/$dataset/test"
train_dir="data/$dataset/train"
model_dir="model/$dataset"

if [[ -d "$img_dir" ]]
then 
	echo "$img_dir already exists"
else
    mkdir "$img_dir"
fi

for fo in $test_dir/*
do
 	for fi in $fo/*
 	do	
 		folder=$(echo $fi | cut -d'/' -f 1,2,3,4)
 		filename=$(echo $fi | cut -d'/' -f 3,4,5)
 		filename=$(echo $filename | tr '/' '_')
 		new_file=$img_dir/$filename
 		cp "$fi" "$new_file"
 	done
done

echo "Moved test images into $img_dir"

for fo in $train_dir/*
do
 	for fi in $fo/*
 	do	
 		folder=$(echo $fi | cut -d'/' -f 1,2,3,4)
 		filename=$(echo $fi | cut -d'/' -f 3,4,5)
 		filename=$(echo $filename | tr '/' '_')
 		new_file=$img_dir/$filename
 		cp "$fi" "$new_file"
 	done
done

echo "Moved train images into $img_dir"

python3 mvtec_dataset.py "$dataset"
echo "CSV file built"

if [[ -d "$model_dir" ]]
then 
	echo "$model_dir already exists"
else
    mkdir "$model_dir"
fi
