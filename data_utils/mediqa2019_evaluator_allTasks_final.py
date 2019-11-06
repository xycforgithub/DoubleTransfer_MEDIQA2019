# From the original file example_evaluator.py by Sharada Mohanty (https://github.com/AICrowd/aicrowd-example-evaluator) 
# Adapted for MEDIQA 2019 by Asma Ben Abacha --Accuracy for Tasks 1 and 2 (NLI and RQE) & MRR, Accuracy, Precision, and Spearman's rank correlation coefficient.
# Last update on April 16, 2019. 

import pdb
import pandas as pd
import numpy as np
import scipy
import scipy.stats
from collections import defaultdict

class MediqaEvaluator:
    def __init__(self, answer_file_path, task=1, round=1):
        """
        `round` : Holds the round for which the evaluation is being done. 
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_file_path = answer_file_path
        self.round = round
        self.task = task

    def _evaluate(self, client_payload, _context={}):
        if self.task == 1:
            return self._evaluate_task_1(client_payload, _context)
        elif self.task == 2:
            return self._evaluate_task_2(client_payload, _context)
        elif self.task == 3:
            return self._evaluate_task_3(client_payload, _context)


    def _evaluate_task_1(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]

        # Result file format: pair_id,label (csv file) 

        col_names = ['pair_id', 'label']

        submission = pd.read_csv(submission_file_path, header=None, names=col_names)
        gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names)

        # Drop duplicates except for the first occurrence.
        submission = submission.drop_duplicates(['pair_id'])

        submission.label = submission.label.astype(str)
        gold_truth.label = gold_truth.label.astype(str)

        submission['entry'] = submission.apply(lambda x: '_'.join(x), axis=1)
        gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(x), axis=1)

        s1 = submission[submission['entry'].isin(gold_truth['entry'])]

        accuracy = s1.size / gold_truth.size

        _result_object = {
            "score": accuracy,
            "score_secondary" : 0.0
        }
        return _result_object

    def _evaluate_task_2(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]

        # Result file format: pair_id,label (csv file) 

        col_names = ['pair_id', 'label']

        submission = pd.read_csv(submission_file_path, header=None, names=col_names, dtype={'pair_id': str, "label": str})
        gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names, dtype={'pair_id': str, "label": str})

        # Drop duplicates except for the first occurrence.
        submission = submission.drop_duplicates(['pair_id'])

        submission.label = submission.label.astype(str)
        gold_truth.label = gold_truth.label.astype(str)

        submission['entry'] = submission.apply(lambda x: '_'.join(x), axis=1)
        gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(x), axis=1)

        s1 = submission[submission['entry'].isin(gold_truth['entry'])]

        accuracy = s1.size / gold_truth.size

        _result_object = {
            "score": accuracy,
            "score_secondary" : 0.0
        }

        return _result_object

    def _evaluate_task_3(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]

        # Result file format: q_id,a_id,label{0/1} 

        col_names = ['question_id','answer_id', 'label']

        submission = pd.read_csv(submission_file_path, header=None, names=col_names)
        gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names)

        # Drop duplicates except for the first occurrence.
        submission = submission.drop_duplicates(['question_id', 'answer_id'])

        submission.label = submission.label.astype(str)
        gold_truth.label = gold_truth.label.astype(str)

        submission['entry'] = submission.apply(lambda x: '_'.join(map(str,x)), axis=1)
        gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(map(str,x)), axis=1)

        s1 = submission[submission['entry'].isin(gold_truth['entry'])]

        accuracy = s1.size / gold_truth.size


        question_ids = []
        correct_answers = {}
        for index, row in gold_truth.iterrows():
            qid = row['question_id']

            if qid not in question_ids:
                question_ids.append(qid)

            if row['label'] == '1':
                if qid not in correct_answers:
                    correct_answers[qid] = []

                correct_answers[qid].append(row['answer_id'])


        P1 = 0.
        P5 = 0.
        P10 = 0.
        spearman = 0.
        pv = 0.
        ref_sizeAt5 = 0.
        ref_sizeAt10 = 0.
        mrr = 0.
        sp_nan_ignoredQs = 0

        for qid in question_ids:
            submitted_correct_answers = []
            if qid not in correct_answers:
                sp_nan_ignoredQs+=1
                continue
            index = 1

            first = True

            for _, row in submission[submission['question_id']==qid].iterrows():
                aid = row['answer_id']
                if row['label'] == '1':
                    if first:
                        mrr += 1. / index
                        first=False

                    if aid in correct_answers[qid]:
                        submitted_correct_answers.append(aid)

                        if index == 1:
                            P1 += 1
                        if index <= 5:
                            P5 += 1
                        if index <= 10:
                            P10 += 1

                index += 1

            matched_gold_subset = []

            for x in correct_answers[qid]:
                if x in submitted_correct_answers:
                    matched_gold_subset.append(x)

            rho, p_value = scipy.stats.spearmanr(submitted_correct_answers, matched_gold_subset)
            if np.isnan(rho):
                rho = 0.0
                sp_nan_ignoredQs += 1
            spearman += rho
            pv += p_value
            ref_sizeAt5 += min(5, len(correct_answers[qid]))
            ref_sizeAt10 += min(10, len(correct_answers[qid]))


        question_nb = len(question_ids)
        q_nb_spearman = question_nb - sp_nan_ignoredQs
        spearman = spearman / q_nb_spearman if q_nb_spearman!=0 else 0.0
        P1 = P1 / question_nb
        if ref_sizeAt5 != 0:
            P5 = P5 / ref_sizeAt5
        else:
            P5 = 0.
        if ref_sizeAt10 !=0:
            P10 = P10 / ref_sizeAt10
        else:
            P10 = 0.
        # print(mrr, question_nb)
        if question_nb !=0:
            mrr = mrr / question_nb
        else:
            mrr = 0.

        if np.isnan(spearman):
            spearman = 0.0

        _result_object = {
            "score": accuracy,
            "score_secondary": spearman,
            "meta" : {
                "MRR": mrr,
                "P@1": P1,
                "P@5": P5,
                "P@10": P10
            }
        }
        return _result_object

def load_qa_pred(pred_path='/pylon5/db5fp9p/yichongx/data/mediqa/task3_qa/gt_dev.csv'):
    pred_dict=defaultdict(list)
    headline=False
    with open(pred_path) as f:
        for line in f:
            if headline:
                headline=False
                continue
            qid,aid,label = line.split(',')
            # if '102' in aid:
            #     print(line, qid, aid, label)
            # uid = '{}____{}'.format(qid,aid)
            label=int(label)
            pred_dict[qid].append((aid,label))
            # pred_dict[uid]=label
    return pred_dict

def eval_mediqa_official(pred_path, ground_truth_path='../data/mediqa/task3_qa/gt_dev.csv', task=3, eval_qa_more=False):
    _client_payload = {}
    _client_payload["submission_file_path"] = pred_path
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = MediqaEvaluator(ground_truth_path, task=task)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context) 

    if task==3 and eval_qa_more:
        pred_dict = load_qa_pred(pred_path)
        gt_dict = load_qa_pred(ground_truth_path)
        cnt=0
        feed_list = []
        for qid in gt_dict:
            pred_scores = {}
            for rank, (aid,pred_label) in enumerate(pred_dict[qid]):
                pred_scores[aid]=100-rank
            this_list=[]
            # if 1 not in [item[1] for item in gt_dict[qid]]:
            #     pdb.set_trace()
            for aid,gt_label in gt_dict[qid]:
                this_list.append((gt_label,pred_scores[aid]))
            feed_list.append(this_list)

        maps, mrrs, pa1s = get_score(feed_list)

        # result['MRR']=mrrs
        result['MAP']=maps
        # result['P@1']=pa1s

    return result

def get_score(total_list):
    correct, wrong = 0, 0 # for p@1
    pred = [] # for MAP
    inv_rank = [] # for MRR
    for this_list in total_list:
        # this_list is a list of tuples (y, yp)
        ys = [l[0] for l in this_list]  # true answers
        yps = [l[1] for l in this_list] # prob of true answer
        if not 1 in ys: 
            # print(this_list)
            continue    # remove cases of no answer
                    # following previous works
        my_preds = [yp for (y, yp) in zip(ys, yps) if y==1]

        yps.sort(reverse=True)

        rank = len(yps)
        for i in my_preds:
            if rank>yps.index(i): rank=yps.index(i)
        rank += 1           # model set groundtruth which rank

        inv_rank.append(1.0/float(rank))# for MRR
        if rank==1: correct+=1      
        else: wrong += 1        # for P@1

        precs = []
        for i, ypi in enumerate(yps):
            if ypi in my_preds:
                prec = (1.0+len(precs))/(i+1.0)
                precs.append(prec)
        if len(precs)==0: pred.append(0.0)
        else: pred.append(np.mean(precs))

    MAP = np.mean(pred)*100
    # print(np.sum(inv_rank), len(inv_rank))
    MRR = np.mean(inv_rank)*100
    P1 = float(correct)/float(correct+wrong)*100
    return (MAP, MRR, P1)    

if __name__ == "__main__":
    task=2
    print("Testing Task (Round-1) : {}".format(task))
    answer_file_path = '../../data/mediqa/task2_rqe/gt_dev.csv'
    _client_payload = {}
    # _client_payload["submission_file_path"] = "/pylon5/db5fp9p/yichongx/model_data/mt_dnn_mediqa/scibert_predict/mediqa_dev_scores_0.csv"
    _client_payload["submission_file_path"] = "../../tmp/my_pred_mediqa/task2/rqe_dev_scores_1.csv"

    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    # aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=task)
    # Evaluate
    result = eval_mediqa_official(_client_payload["submission_file_path"], answer_file_path, task, True)
    print(result)

    # # Test Tasks 1,2,3
    # # for task in range(1, 4):
    # task=2
    # print("Testing Task (Round-1) : {}".format(task))
    # # answer_file_path = "/pylon5/db5fp9p/yichongx/data/mediqa/task3_qa/gt_train.csv"
    # # answer_file_path = "../data/task{}/ground_truth.csv".format(task)
    # answer_file_path = '../../data/mediqa/task2_rqe/gt_test_my.csv'
    # _client_payload = {}
    # # _client_payload["submission_file_path"] = "/pylon5/db5fp9p/yichongx/model_data/mt_dnn_mediqa/scibert_predict/mediqa_dev_scores_0.csv"
    # _client_payload["submission_file_path"] = "../../tmp/my_pred_mediqa/task2/rqe_test_scores_3.csv"

    # # Instaiate a dummy context
    # _context = {}
    # # Instantiate an evaluator
    # aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=task)
    # # Evaluate
    # result = aicrowd_evaluator._evaluate(_client_payload, _context)
    # print(result)

    # # Test Tasks 1,2,3 - Round -2
    # for task in range(1, 4):
    #     print("Testing Task (Round-2) : {}".format(task))
    #     answer_file_path = "data/task{}/ground_truth_round_2.csv".format(task)
    #     _client_payload = {}
    #     _client_payload["submission_file_path"] = "data/task{}/sample_submission_round_2.csv".format(task)

    #     # Instaiate a dummy context
    #     _context = {}
    #     # Instantiate an evaluator
    #     aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=task, round=2)
    #     # Evaluate
    #     result = aicrowd_evaluator._evaluate(_client_payload, _context)
    #     print(result)
