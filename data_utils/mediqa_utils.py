import json
import numpy as np
from random import shuffle,sample
import random
from .label_map import METRIC_FUNC, METRIC_META, METRIC_NAME,GLOBAL_MAP
import xml.etree.ElementTree as ET 
import os
import pdb
from collections import defaultdict
import re

test_mode=False
mediqa_name_list=['mediqa','mediqa_url']
def load_mednli(file, label_dict):
	rows = []
	cnt = 0
	with open(file, encoding="utf8") as f:
		for line_raw in f:
			data_line = json.loads(line_raw)
			lab = label_dict[data_line['gold_label']]
			if lab is None:
				if data_line['gold_label']=='':
					lab=0
				else:
					import pdb; pdb.set_trace()
			lab = 0 if lab is None else lab
			sample = {'uid': data_line['pairID'], 'premise': data_line['sentence1'], 'hypothesis': data_line['sentence2'], 'label': lab}
			rows.append(sample)
			cnt += 1
			if test_mode and cnt==200:
				print('use first 200 samples for test.')
				break
	return rows

def load_rqe(file, label_dict):
	rows = []
	cnt = 0
	tree = ET.parse(file)
	root = tree.getroot()
	testset = 'TestSet' in file
	for xml_sample in root.findall('./pair'):
		if 'value' not in xml_sample.attrib:
			assert testset
			this_label = 0
		else:
			this_label = label_dict[xml_sample.attrib['value']]
		sample={'label': this_label, 'uid': xml_sample.attrib['pid']}
		if sample['label'] is None:
			import pdb; pdb.set_trace()
		for child in xml_sample:
			if child.tag=='chq':
				assert 'premise' not in sample
				sample['premise']=child.text
			if child.tag=='faq':
				assert 'hypothesis' not in sample
				sample['hypothesis']=child.text
		rows.append(sample)
		cnt += 1
		if test_mode and cnt==200:
			print('use first 200 samples for test.')
			break
	return rows

def shuffle_rqe(train_data, dev_data, test_data):
	random.seed(1)
	hyp_dict = defaultdict(list)
	for sample in dev_data:
		hyp_dict[sample['hypothesis']].append(sample)
	new_train_data = [sample for sample in train_data]
	new_dev_data = []
	for hyp in hyp_dict:
		p=random.random()
		if p>0.5:
			for sample in hyp_dict[hyp]:
				new_train_data.append(sample)
			# if len(hyp_dict[hyp])>1:
				# print('len=',len(hyp_dict[hyp]),':',hyp)
		else:
			for sample in hyp_dict[hyp]:
				new_dev_data.append(sample)
			# if len(hyp_dict[hyp])>1:
				# print('len=',len(hyp_dict[hyp]),':',hyp)
	print('splitted dev size:',len(new_dev_data))
	# pdb.set_trace()
	my_test_path = '../data/mediqa/task2_rqe/gt_test_my.csv'
	header=True
	my_test_ids=[]
	my_test_labels={}
	with open(my_test_path) as f:
		for line in f:
			if header:
				header=False
				continue
			test_id, test_label = line.split(',')
			my_test_ids.append(test_id)
			my_test_labels[test_id]=int(test_label)
	part_test_data = []
	for sample in test_data:
		if sample['uid'] in my_test_ids:
			new_sample={k:v for k,v in sample.items()}
			new_sample['uid']='my_{}'.format(sample['uid'])
			new_sample['label']=my_test_labels[sample['uid']]
			part_test_data.append(new_sample)
	print('length part data:', len(part_test_data))
	new_dev_data = new_dev_data + part_test_data
	# pdb.set_trace()
	return new_train_data, new_dev_data, part_test_data, test_data




				


def load_mediqa(dir_path,label_dict, add_url_words):
	# if is_train and not test_mode:
	fn_train1=os.path.join(dir_path,'MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml')
	fn_train2=os.path.join(dir_path,'MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml')
	fn_dev=os.path.join(dir_path,'MEDIQA2019-Task3-QA-ValidationSet.xml')
	fn_test=os.path.join(dir_path,'MEDIQA_Task3_QA_TestSet.xml')

	train1_rows, train1_dict, train1_goods = load_mediqa_file(fn_train1, 'train1', add_url_words)
	train2_rows, train2_dict, train2_goods = load_mediqa_file(fn_train2, 'train2', add_url_words)
	dev_rows, dev_dict, dev_goods = load_mediqa_file(fn_dev, None, add_url_words)
	test_rows, _, _ = load_mediqa_file(fn_test, None, add_url_words)

	# shuffle train/dev data to contain both alexa and livemed
	valid_ids = dev_goods[-25:]+train2_goods[-25:] # dev is all livemed
	train_n_dev_dict = {**train1_dict, **train2_dict, **dev_dict} # merge the dicts
	new_train_rows = [v for k,v in train_n_dev_dict.items() if k not in valid_ids]
	new_train_rows = sum(new_train_rows, []) # expand the list
	new_dev_rows = [v for k,v in train_n_dev_dict.items() if k in valid_ids]
	new_dev_rows = sum(new_dev_rows, [])
	# vid = train2_goods[-30:]
	# new2_dev_rows = [v for k,v in train_n_dev_dict.items() if k in vid]
	# new2_dev_rows = sum(new2_dev_rows, [])
	# print(len(new2_dev_rows))
	# pdb.set_trace()

	splits = 5 # cross validation
	must_dev = dev_goods[-12:]+train2_goods[-12:]
	must_dev_rows = [v for k,v in train_n_dev_dict.items() if k in must_dev]
	must_dev_rows = sum(must_dev_rows, [])	
	print('must dev rows:',len(must_dev_rows))
	split_pool = [key for key in train_n_dev_dict.keys() if key not in must_dev]

	key_pools=[set() for i in range(splits)]
	cnter=0
	for key in split_pool:
		key_pools[cnter].add(key)
		cnter = (cnter+1)%splits
	split_datas = [[] for i in range(splits)]
	for sidx in range(splits):
		for key in key_pools[sidx]:
			split_datas[sidx].append(train_n_dev_dict[key])

	cv_datas=[]
	for sidx in range(splits):
		train_data=[]
		dev_data = split_datas[sidx]
		for sidx2 in range(splits):
			if sidx2!=sidx:
				train_data+=split_datas[sidx2]
		train_data = sum(train_data, [])
		dev_data = sum(dev_data, [])
		old_dev_len = len(dev_data)
		dev_data+=must_dev_rows
		print('lengths:',len(train_data), old_dev_len, len(dev_data))
		cv_datas.append((train_data,dev_data))
	# pdb.set_trace()

	old_dev_rows = dev_rows

	return new_train_rows, new_dev_rows, old_dev_rows, test_rows, cv_datas

def checkdigit(w):
	return sum(c.isdigit() for c in w)

def load_mediqa_file(fn,f_idx=None, add_url_words=False):
	rows = []
	cnt = 0
	possible_scores=[4,3,2,1]
	tree = ET.parse(fn)
	root = tree.getroot()
	exclude_words_url = ['htm','html','','https','www']
	url_words = set()
	question_dict=defaultdict(list)
	good_questions=[]
	testset = 'TestSet' in fn
	for xml_sample in root.findall('./Question'):
		# sample={'label': label_dict[xml_sample.attrib['value']], 'uid': xml_sample.attrib['pid']}
		# if sample['label'] is None:
		# 	import pdb; pdb.set_trace()
		qid=xml_sample.attrib['QID']
		this_qid='f{}_{}'.format(f_idx,qid) if f_idx is not None else qid

		question_text = xml_sample.find('QuestionText').text
		answers=[]
		current_cnt=cnt
		all_scores=[]
		all_ranks=[]
		all_labels=[]
		for ans_node in xml_sample.find('AnswerList'):
			answer_text = ans_node.find('AnswerText').text
			if add_url_words:
				answer_url = ans_node.find('AnswerURL').text
				# try:
				answer_url = re.split('\W+',answer_url)
				# except:
				# 	pdb.set_trace()
				answer_url = [w for w in answer_url if w not in exclude_words_url and not checkdigit(w)]
				url_words.update(answer_url)
				answer_text = ' '.join(answer_url+['[PAD]', answer_text])

			if 'ReferenceScore' not in ans_node.attrib:
				assert testset
				this_score='1'
				this_rank='0'
			else:
				this_score=ans_node.attrib['ReferenceScore']
				this_rank = ans_node.attrib['ReferenceRank']
			all_scores.append(this_score)
			all_ranks.append(this_rank)
			all_labels.append(-1)
			this_uid =  '{}____{}'.format(this_qid,ans_node.attrib['AID'])
			rows.append({'uid':this_uid,'premise':answer_text,
				'hypothesis':question_text,'score':this_score,'rank':this_rank})
			question_dict[this_qid].append(rows[-1])
			cnt += 1
			# pdb.set_trace()
		first_rank=1
		has_good = False
		for sc in possible_scores:
			num_this_score=all_scores.count(str(sc))
			if num_this_score==0:
				continue
			step_size=1.0/num_this_score
			for idx,(ts,tr) in enumerate(zip(all_scores,all_ranks)):
				if int(ts)==sc:
					all_labels[idx]=sc - step_size*(int(tr) - first_rank)
					assert int(tr)<=len(all_ranks)
					if sc>=3:
						has_good = True
					# if testset:
					# 	pdb.set_trace()

			first_rank+=num_this_score
		if has_good:
			good_questions.append(this_qid)


		# print(all_scores,all_ranks,all_labels)
		# pdb.set_trace()
		for idx in range(current_cnt,cnt):
			rows[idx]['label']=all_labels[idx - current_cnt]
			# try:
			assert rows[idx]['label']!=-1
			# except:
			# 	print('dsfs')
			# 	pdb.set_trace()

		# if test_mode and cnt>=250:
		# 	print('use first 250 samples for test mediqa.')
		# 	break
	# print('url_words:',url_words)
	# if testset:
	# 	pdb.set_trace()

	return rows, question_dict, good_questions
	
def load_medquad(dir_path,negative_num=1,ratio=0.9, random_seed=0):
	train_rows=[]
	valid_rows=[]
	random.seed(random_seed)
	cnt=0
	if test_mode:
		ratio=0.5
	pos_cnt=0
	neg_cnt=0
	for subset_dir in os.listdir(dir_path):
		this_path = os.path.join(dir_path,subset_dir)
		if not os.path.isdir(this_path) or subset_dir=='empty_sets':
			continue
		for fn in os.listdir(this_path):
			p = random.random()
			rows = train_rows if p<ratio else valid_rows
			qids=[]
			questions=[]
			answers=[]
			try:
				tree = ET.parse(os.path.join(this_path, fn))
			except:
				pdb.set_trace()
			root = tree.getroot()
			pairs = root.find('QAPairs')
			question_answer_dict = {}
			Q_IND='Question'
			A_IND='Answer'
			P_IND='QAPair'
			if pairs is None:
				pairs = root.find('qaPairs')
				Q_IND='question'
				A_IND='answer'
				P_IND='pair'
				# pdb.set_trace()
			pairlist=pairs.findall(P_IND)
			if len(pairlist)==0:
				print('no pair:',this_path,fn)
				# pdb.set_trace()
			for qapair in pairlist:
				try:
					qid = qapair.find(Q_IND).attrib['qid']
					qtext = qapair.find(Q_IND).text
					atext = qapair.find(A_IND).text
				except:
					pdb.set_trace()
				# if qid=='0000223-4':
				# 	pdb.set_trace()
				if qtext is None or qid is None:
					pdb.set_trace()
				if atext is None:
					print('skip because none answer:',qid,this_path,fn)
					continue

				qids.append(qid)
				questions.append(qtext)
				answers.append(atext)
				rows.append({'uid':qid,'premise':atext,'hypothesis':qtext,'label':1})
				if qtext not in question_answer_dict:
					question_answer_dict[qtext]=set([atext])
				else:
					question_answer_dict[qtext].add(atext)
				cnt+=1
				pos_cnt+=1
			aset=set(answers)
			for qid,qtext,atext in zip(qids,questions,answers):
				aset-=question_answer_dict[qtext]
				try:
					assert atext not in aset
				except:
					pdb.set_trace()
				if len(aset)==0:
					# print('skip negative:',subset_dir,fn)
					continue
				fake_answers = random.sample(aset,min(negative_num, len(aset)))
				for fidx,fatext in enumerate(fake_answers):
					rows.append({'uid':qid+'_neg_{}'.format(fidx),'premise':fatext,
						'hypothesis':qtext,'label':0})
					neg_cnt+=1
				aset|=question_answer_dict[qtext]

			if test_mode and cnt>=200:
				break
	print('positive:',pos_cnt,'negative:',neg_cnt)
	if test_mode:
		return valid_rows,valid_rows
	else:
		return train_rows, valid_rows

def submit(path, data, dataset_name, label_dict=None, 
	threshold=2.0, mediqa_mustanswer=False, mednli_comb=True):
	if dataset_name == 'mednli' and mednli_comb:
		header='pair_id,label'
		label_dict = GLOBAL_MAP['mednli']
		with open(path ,'w') as writer:
			writer.write('{}\n'.format(header))
			predictions, uids, scores = data['predictions'], data['uids'], data['scores']
			assert len(predictions) == len(uids)
			scores = np.array(scores).reshape(-1,3)
			assert len(uids) % 3 == 0
			possible_combs=[(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
			paired=[]
			for idx in range(0, len(uids), 3):
				max_prob=-100.0
				max_comb = 0
				for cid,(id1,id2,id3) in enumerate(possible_combs):
					this_prob = np.log(scores[idx,id1])+np.log(scores[idx+1,id2])+np.log(scores[idx+2,id3])
					if this_prob>max_prob:
						max_prob = this_prob
						max_comb = cid
				# pdb.set_trace()
				paired.append((uids[idx],possible_combs[max_comb][0]))
				paired.append((uids[idx+1],possible_combs[max_comb][1]))
				paired.append((uids[idx+2],possible_combs[max_comb][2]))

			for uid, pred in paired:
				this_pred=pred
				if label_dict is not None:
					this_pred=label_dict[this_pred]
				writer.write('{},{}\n'.format(uid, this_pred))

	elif dataset_name in ['mednli', 'rqe','medquad']:
		header='pair_id,label'
		if dataset_name =='mednli':
			label_dict = GLOBAL_MAP['mednli']
		with open(path ,'w') as writer:
			writer.write('{}\n'.format(header))
			predictions, uids = data['predictions'], data['uids']
			assert len(predictions) == len(uids)
			# sort label
			paired = [(uid, predictions[idx]) for idx, uid in enumerate(uids)]
			# paired = sorted(paired, key=lambda item: item[0])
			for uid, pred in paired:
				this_pred=pred
				if label_dict is not None:
					this_pred=label_dict[this_pred]
				writer.write('{},{}\n'.format(uid, this_pred))
					
	else:
		assert dataset_name in mediqa_name_list
		scores = np.array(data['scores'])
		q_map=defaultdict(list)
		for uid,score in zip(data['uids'],scores):
			try:
				qid,aid=uid.split('____')
			except:
				pdb.set_trace()
			q_map[qid].append((aid,score))
		with open(path ,'w') as writer:
			for qid in q_map:
				ranked_list=sorted(q_map[qid],key=lambda x: x[1], reverse=True)
				first_answer = mediqa_mustanswer
				for aid,score in ranked_list:
					pred=1 if score>threshold else 0
					if first_answer:
						pred=1
						first_answer=False
					writer.write('{},{},{}\n'.format(qid, aid, pred))

def eval_model(model, data, dataset, use_cuda=True, with_label=True):
	data.reset()
	if use_cuda:
		model.cuda()
	predictions = []
	golds = []
	scores = []
	ids = []
	metrics = {}
	for idx, (batch_meta, batch_data) in enumerate(data):
		score, pred, gold = model.predict(batch_meta, batch_data)
		predictions.extend(pred)
		golds.extend(gold)
		scores.extend(score)
		ids.extend(batch_meta['uids'])
	mmeta = METRIC_META[dataset]
	# pdb.set_trace()
	if with_label:
		for mm in mmeta:
			metric_name = METRIC_NAME[mm]
			metric_func = METRIC_FUNC[mm]
			# if mm < 3:
			metric = metric_func(predictions, golds)
			# else:
				# metric = metric_func(scores, golds)
			metrics[metric_name] = metric
	return metrics, predictions, scores, golds, ids
