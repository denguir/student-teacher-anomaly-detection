#!/bin/bash

DATA_URL="ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"
DATA_DIR="data"
MODEL_DIR="model"

CATEGORIES=(bottle
			cable
			capsule
			carpet
			grid
			hazelnut
			leather
			metal_nut
			pill
			screw
			tile
			toothbrush
			transistor
			wood
			zipper
			)


function log {
    local PURPLE='\033[0;35m'
    local NOCOLOR='\033[m'
    local BOLD='\033[1m'
    local NOBOLD='\033[0m'
    echo -e -n "${PURPLE}${BOLD}$1${NOBOLD}${NOCOLOR}"
}

function prepare_dir {
	mkdir -p $DATA_DIR
	mkdir -p $MODEL_DIR
}

function download_dataset {
	log "Downloading MVTec dataset...\\n"
	wget -nc $DATA_URL -P $DATA_DIR
	log "Done!\\n"
}

function extract_dataset {
	log "Extracting MVTec dataset...\\n"
	tar -xf "$DATA_DIR/mvtec_anomaly_detection.tar.xz" -C $DATA_DIR
	rm -rf "$DATA_DIR/mvtec_anomaly_detection.tar.xz"
	chmod -R u+rw $DATA_DIR
	log "Done!\\n"
}

function move_images {
	DATASET="$1"
	SRC_DIR="$DATA_DIR/$DATASET/$2"
	TGT_DIR="$DATA_DIR/$DATASET/$3"
	mkdir -p $TGT_DIR
	for fo in $SRC_DIR/*
	do
		for fi in $fo/*
		do	
			folder=$(echo $fi | cut -d'/' -f 1,2,3,4)
			filename=$(echo $fi | cut -d'/' -f 3,4,5)
			filename=$(echo $filename | tr '/' '_')
			dest_file=$TGT_DIR/$filename
			mv $fi $dest_file
		done
	done
}

function build_csv {
	DATASET="$1"
	python3 mvtec_dataset.py $DATASET
}


function process_dataset {
	log "Processing MVTec dataset...\\n"
	TEST_DIR="test"
	TRAIN_DIR="train"
	GT_DIR="ground_truth"
	IMG_DIR="img"
	for CAT in "${CATEGORIES[@]}"
	do 
		move_images $CAT $TEST_DIR $IMG_DIR
		move_images $CAT $TRAIN_DIR $IMG_DIR
		move_images $CAT $GT_DIR $GT_DIR
		build_csv $CAT
	done
	log "Done!\\n"
}

prepare_dir
download_dataset
extract_dataset
process_dataset
