#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for MT-DNN experiments
############################################################## 

DATA_DIR=$(pwd)/../data/glue_data/
echo "Create a folder $DATA_DIR"
mkdir ${DATA_DIR}

BERT_DIR=$(pwd)/../bert_models
echo "Create a folder BERT_DIR"
mkdir ${BERT_DIR}

## DOWNLOAD MNLI DATA
## Please refer glue-baseline install requirments or other issues.
git clone https://github.com/nyu-mll/jiant.git
cd jiant
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks MNLI
cd ..
rm -rf jiant

## Download data from MEDIQA website and put in ../data/mediqa. Due to privacy restrictions we cannot release data publicly.

########################

cd ${BERT_DIR}
## DOWNLOAD BERT vocabulary
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O "uncased_bert_base.zip"
unzip uncased_bert_base.zip
mv uncased_L-12_H-768_A-12/vocab.txt "${BERT_DIR}/"
rm *.zip
rm -rf uncased_L-12_H-768_A-12

## Download MT-DNN models
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt -O "${BERT_DIR}/mt_dnn_base.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt -O "${BERT_DIR}/mt_dnn_large.pt"

## Download SciBERT models
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar -O "uncased_scibert_base.tar"
tar -xvf uncased_scibert_base.tar
