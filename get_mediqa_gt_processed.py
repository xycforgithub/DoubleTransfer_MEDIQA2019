import xml.etree.ElementTree as ET
from data_utils.mediqa_utils import submit
import os
import pdb
import json

is_train=False

dir_path='../data/mediqa/task1_mednli/'
processed_path = '../data/mediqa_processed/mt_dnn_mediqa_scibert_v2/'
if is_train:
	dev_path=os.path.join(processed_path,'mednli_train.json')
else:
	dev_path=os.path.join(processed_path,'mednli_dev.json')
uids=[]
preds=[]
with open(dev_path,encoding='utf-8') as f:
	for line in f:
		sample=json.loads(line)
		uids.append(sample['uid'])
		preds.append(sample['label'])
output_path=os.path.join(dir_path,'gt_train.csv') if is_train else os.path.join(dir_path,'gt_dev.csv')
result={'uids':uids,'predictions':preds}
submit(output_path, result, 'mednli')

dir_path='../data/mediqa/task2_rqe/'
processed_path = '../data/mediqa_processed/mt_dnn_mediqa_scibert_v2/'
if is_train:
	dev_path=os.path.join(processed_path,'rqe_train.json')
else:
	dev_path=os.path.join(processed_path,'rqe_dev.json')
uids=[]
preds=[]
with open(dev_path,encoding='utf-8') as f:
	for line in f:
		sample=json.loads(line)
		uids.append(sample['uid'])
		preds.append(sample['label'])
output_path=os.path.join(dir_path,'gt_train.csv') if is_train else os.path.join(dir_path,'gt_dev.csv')
result={'uids':uids,'predictions':preds}
submit(output_path, result, 'rqe')

dataset_name='rqe_shuff'
dir_path='../data/mediqa/task2_rqe/'
processed_path = '../data/mediqa_processed/mt_dnn_mediqa_scibert_v2/'
if is_train:
	dev_path=os.path.join(processed_path,'{}_train.json'.format(dataset_name))
else:
	dev_path=os.path.join(processed_path,'{}_dev.json'.format(dataset_name))
uids=[]
preds=[]
with open(dev_path,encoding='utf-8') as f:
	for line in f:
		sample=json.loads(line)
		uids.append(sample['uid'])
		preds.append(sample['label'])
output_path=os.path.join(dir_path,'gt_train_{}.csv'.format(dataset_name)) if is_train else os.path.join(dir_path,'gt_dev_{}.csv'.format(dataset_name))
result={'uids':uids,'predictions':preds}
submit(output_path, result, 'rqe')

dir_path='../data/mediqa/MedQuAD/'
processed_path = '../data/mediqa_processed/mt_dnn_mediqa_scibert_v2/'
if is_train:
	dev_path=os.path.join(processed_path,'medquad_train.json')
else:
	dev_path=os.path.join(processed_path,'medquad_dev.json')
uids=[]
preds=[]
with open(dev_path,encoding='utf-8') as f:
	for line in f:
		sample=json.loads(line)
		uids.append(sample['uid'])
		preds.append(sample['label'])
output_path=os.path.join(dir_path,'gt_train.csv') if is_train else os.path.join(dir_path,'gt_dev.csv')
result={'uids':uids,'predictions':preds}

submit(output_path, result, 'medquad')

dir_path='../data/mediqa/task3_qa/'
processed_path = '../data/mediqa_processed/mt_dnn_mediqa_scibert_v2/'
if is_train:
	dev_path=os.path.join(processed_path,'mediqa_train.json')
else:
	dev_path=os.path.join(processed_path,'mediqa_dev.json')
uids=[]
scores=[]
with open(dev_path,encoding='utf-8') as f:
	for line in f:
		sample=json.loads(line)
		uids.append(sample['uid'])
		scores.append(sample['label'])
output_path=os.path.join(dir_path,'gt_train.csv') if is_train else os.path.join(dir_path,'gt_dev.csv')
result={'uids':uids,'scores':scores}
submit(output_path, result, 'mediqa', threshold=2.000001)

dir_path='../data/mediqa/task3_qa/'
processed_path = '../data/mediqa_processed/mt_dnn_mediqa_scibert_v2/'
for sidx in range(0,5):
	if is_train:
		dev_path=os.path.join(processed_path,'mediqa_{}_train.json'.format(sidx))
	else:
		dev_path=os.path.join(processed_path,'mediqa_{}_dev.json'.format(sidx))
	uids=[]
	scores=[]
	with open(dev_path,encoding='utf-8') as f:
		for line in f:
			sample=json.loads(line)
			uids.append(sample['uid'])
			scores.append(sample['label'])
	output_path=os.path.join(dir_path,'gt_train_{}.csv'.format(sidx)) if is_train else os.path.join(dir_path,'gt_dev_{}.csv'.format(sidx))
	result={'uids':uids,'scores':scores}
	submit(output_path, result, 'mediqa', threshold=2.000001)

