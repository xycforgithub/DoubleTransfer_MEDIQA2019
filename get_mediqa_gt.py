import xml.etree.ElementTree as ET
from data_utils.mediqa_utils import submit
import os
import pdb

label_corres={'4':1.0,'3':0.75,'2':0.25,'1':0.0}
dir_path='../data/mediqa/task3_qa/'
is_train=True
if is_train:
	fn1=os.path.join(dir_path,'MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml')
	fn2=os.path.join(dir_path,'MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml')
	fn_list=[fn1,fn2]
else:
	fn_list=[os.path.join(dir_path,'MEDIQA2019-Task3-QA-ValidationSet.xml')]


uids=[]
scores=[]
for f_idx, dev_path in enumerate(fn_list):
	tree = ET.parse(dev_path)
	root = tree.getroot()
	for xml_sample in root.findall('./Question'):
		# sample={'label': label_dict[xml_sample.attrib['value']], 'uid': xml_sample.attrib['pid']}
		# if sample['label'] is None:
		# 	import pdb; pdb.set_trace()
		qid=xml_sample.attrib['QID']
		question_text = xml_sample.find('QuestionText').text
		answers=[]
		for ans_node in xml_sample.find('AnswerList'):
			answer_text = ans_node.find('AnswerText').text
			aid=ans_node.attrib['AID']
			this_uid = 'f{}_{}____{}'.format(f_idx,qid,ans_node.attrib['AID']) if is_train else '____'.join([qid,ans_node.attrib['AID']])
			uids.append(this_uid)
			this_score=1.0 if int(ans_node.attrib['ReferenceScore']) in [3,4] else 0.5
			this_score-=int(ans_node.attrib['ReferenceRank'])*0.01
			scores.extend([this_score])
			# pdb.set_trace()
output_path=os.path.join(dir_path,'gt_train.csv') if is_train else os.path.join(dir_path,'gt_dev.csv')
result={'uids':uids,'scores':scores}
# pdb.set_trace()
submit(output_path, result, 'mediqa', threshold=0.5)
