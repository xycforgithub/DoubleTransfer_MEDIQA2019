# Copyright (c) Microsoft. All rights reserved.
import os
import json
import tqdm
import pickle
import re
import collections
import argparse
from sys import path
from data_utils.vocab import Vocabulary
from bert.tokenization import BertTokenizer
from data_utils.log_wrapper import create_logger
from data_utils.label_map import GLOBAL_MAP
from data_utils.glue_utils import *
from data_utils.mediqa_utils import *
import pdb

MAX_SEQ_LEN = 384

logger = create_logger(__name__, to_disk=True, log_file='bert_data_proc_{}.log'.format(MAX_SEQ_LEN))

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def build_data(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True, extra_features=[]):
    """Build data of sentence pair tasks
    """
    print(dump_path)
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis = bert_tokenizer.tokenize(sample['hypothesis'])
            label = sample['label']
            original_len_hyp = len(hypothesis)
            _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
            # print('lens:',len(premise),len(hypothesis))
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            for feature_name in extra_features:
                features[feature_name]=sample[feature_name]
            writer.write('{}\n'.format(json.dumps(features)))


def generate_gt_csv(data, dump_path): # now only use for first 2 tasks
    header='pair_id,label'
    with open(dump_path,'w') as writer:
        writer.write('{}\n'.format(header))
        for sample in data:
            uid, label = sample['uid'],sample['label']
            if 'task3_qa' not in dump_path:
                writer.write('{},{}\n'.format(uid, label))
            else:
                qid,aid=uid.split('____')
                writer.write('{},{},{}\n'.format(qid, aid, label))            

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing MEDIQA dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='../data')
    parser.add_argument('--cased', action='store_true', help='cased model.')
    parser.add_argument('--sci_vocab', action='store_true', help='scibert vocab.')
    args = parser.parse_args()
    return args

def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    train_name = 'dev' if test_mode else 'train'

    ######################################
    # MNLI Task
    ######################################
    
    glue_path = os.path.join(root,'glue_data')
    multi_train_path =  os.path.join(glue_path, 'MNLI/train.tsv')
    multi_dev_matched_path = os.path.join(glue_path, 'MNLI/dev_matched.tsv')
    multi_dev_mismatched_path = os.path.join(glue_path, 'MNLI/dev_mismatched.tsv')
    multi_test_matched_path = os.path.join(glue_path, 'MNLI/test_matched.tsv')
    multi_test_mismatched_path = os.path.join(glue_path, 'MNLI/test_mismatched.tsv')

    ######################################
    # MedNLI Task
    ######################################

    mednli_train_path = os.path.join(root, 'mediqa/task1_mednli/mli_{}_v1.jsonl'.format(train_name))
    mednli_dev_path = os.path.join(root, 'mediqa/task1_mednli/mli_dev_v1.jsonl')
    mednli_test_path = os.path.join(root, 'mediqa/task1_mednli/mli_test_v1.jsonl')
    mednli_realtest_path = os.path.join(root, 'mediqa/task1_mednli/mednli_bionlp19_shared_task.jsonl')

    ######################################
    # RQE Task
    ######################################
    rqe_train_path = os.path.join(root, 'mediqa/task2_rqe/RQE_Train_8588_AMIA2016.xml')
    rqe_dev_path = os.path.join(root, 'mediqa/task2_rqe/RQE_Test_302_pairs_AMIA2016.xml')
    rqe_test_path = os.path.join(root, 'mediqa/task2_rqe/MEDIQA_Task2_RQE_TestSet.xml')
    ######################################
    # QA Task
    ######################################

    mediqa_dir = os.path.join(root,'mediqa/task3_qa/')
    ######################################
    # MedQuAD Task
    ######################################

    medquad_dir = os.path.join(root,'mediqa/MedQuAD/')

    
    ######################################
    # Loading DATA
    ######################################

    mednli_train_data = load_mednli(mednli_train_path, GLOBAL_MAP['mednli'])
    mednli_dev_data = load_mednli(mednli_dev_path, GLOBAL_MAP['mednli'])
    mednli_test_data = load_mednli(mednli_test_path, GLOBAL_MAP['mednli'])
    mednli_realtest_data = load_mednli(mednli_realtest_path, GLOBAL_MAP['mednli'])
    mednli_train_data = mednli_train_data + mednli_dev_data
    mednli_dev_data = mednli_test_data
    mednli_test_data = mednli_realtest_data
    logger.info('Loaded {} MedNLI train samples'.format(len(mednli_train_data)))
    logger.info('Loaded {} MedNLI dev samples'.format(len(mednli_dev_data)))
    logger.info('Loaded {} MedNLI test samples'.format(len(mednli_test_data)))

    multinli_train_data = load_mnli(multi_train_path, GLOBAL_MAP['snli'])
    multinli_matched_dev_data = load_mnli(multi_dev_matched_path, GLOBAL_MAP['snli'])
    multinli_mismatched_dev_data = load_mnli(multi_dev_mismatched_path, GLOBAL_MAP['snli'])
    multinli_matched_test_data = load_mnli(multi_test_matched_path, GLOBAL_MAP['snli'], is_train=False)
    multinli_mismatched_test_data = load_mnli(multi_test_mismatched_path, GLOBAL_MAP['snli'], is_train=False)
    logger.info('Loaded {} MNLI train samples'.format(len(multinli_train_data)))
    logger.info('Loaded {} MNLI matched dev samples'.format(len(multinli_matched_dev_data)))
    logger.info('Loaded {} MNLI mismatched dev samples'.format(len(multinli_mismatched_dev_data)))
    logger.info('Loaded {} MNLI matched test samples'.format(len(multinli_matched_test_data)))
    logger.info('Loaded {} MNLI mismatched test samples'.format(len(multinli_mismatched_test_data)))    

    rqe_train_data = load_rqe(rqe_train_path, GLOBAL_MAP['rqe'])
    rqe_dev_data = load_rqe(rqe_dev_path, GLOBAL_MAP['rqe'])
    rqe_test_data = load_rqe(rqe_test_path, GLOBAL_MAP['rqe'])
    logger.info('Loaded {} rqe train samples'.format(len(rqe_train_data)))
    logger.info('Loaded {} rqe dev samples'.format(len(rqe_dev_data)))
    logger.info('Loaded {} rqe test samples'.format(len(rqe_test_data)))

    # shuffled version of RQE data
    rqe_shuff_train_data, rqe_shuff_dev_data, rqe_shuff_test_data = shuffle_rqe(rqe_train_data, rqe_dev_data, rqe_test_data)

    logger.info('Loaded {} rqe_shuff train samples'.format(len(rqe_shuff_train_data)))
    logger.info('Loaded {} rqe_shuff dev samples'.format(len(rqe_shuff_dev_data)))
    logger.info('Loaded {} rqe_shuff test samples'.format(len(rqe_shuff_test_data)))

    mediqa_train_data, mediqa_dev_data, mediqa_old_dev_data, mediqa_test_data, mediqa_split_data = load_mediqa(mediqa_dir,GLOBAL_MAP['mediqa'],False)

    # mediqa_dev_data = load_mediqa(mediqa_dir,GLOBAL_MAP['mediqa'], is_train=False)
    logger.info('Loaded {} mediqa train samples'.format(len(mediqa_train_data)))
    logger.info('Loaded {} mediqa dev samples'.format(len(mediqa_dev_data)))
    logger.info('Loaded {} mediqa test samples'.format(len(mediqa_test_data)))
    # pdb.set_trace()
    for pair in mediqa_split_data:
        train_data,dev_data = pair
        logger.info('Loaded {},{} mediqa splitted samples'.format(len(train_data),len(dev_data)))

    mednli_gt_path = os.path.join(root,'mediqa/task1_mednli/gt_dev.csv')
    generate_gt_csv(mednli_dev_data, mednli_gt_path)
    rqe_gt_path = os.path.join(root,'mediqa/task2_rqe/gt_dev.csv')
    generate_gt_csv(rqe_dev_data, rqe_gt_path)
    mediqa_gt_path = os.path.join(root,'mediqa/task3_qa/gt_dev.csv')
    generate_gt_csv(mediqa_dev_data, mediqa_gt_path)

    add_num=2
    medquad_train_data, medquad_dev_data= load_medquad(medquad_dir, negative_num=add_num)
    logger.info('Loaded {} medquad train samples'.format(len(medquad_train_data)))
    logger.info('Loaded {} medquad dev samples'.format(len(medquad_dev_data)))

    # pdb.set_trace()

    output_name = 'mt_dnn_mediqa'
    if MAX_SEQ_LEN!=512:
        output_name+='_{}'.format(MAX_SEQ_LEN)
    if args.cased:
        output_name+='_cased'
    if args.sci_vocab:
        output_name+='_scibert'
    if test_mode:
        output_name+='_test'

    print('output_name:',output_name)

    mt_dnn_root = os.path.join(root, 'mediqa_processed',output_name)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    # BUILD MNLI
    multinli_train_fout = os.path.join(mt_dnn_root, 'mnli_train.json')
    multinli_matched_dev_fout = os.path.join(mt_dnn_root, 'mnli_matched_dev.json')
    multinli_mismatched_dev_fout = os.path.join(mt_dnn_root, 'mnli_mismatched_dev.json')
    multinli_matched_test_fout = os.path.join(mt_dnn_root, 'mnli_matched_test.json')
    multinli_mismatched_test_fout = os.path.join(mt_dnn_root, 'mnli_mismatched_test.json')
    build_data(multinli_train_data, multinli_train_fout)
    build_data(multinli_matched_dev_data, multinli_matched_dev_fout)
    build_data(multinli_mismatched_dev_data, multinli_mismatched_dev_fout)
    build_data(multinli_matched_test_data, multinli_matched_test_fout)
    build_data(multinli_mismatched_test_data, multinli_mismatched_test_fout)
    logger.info('done with mnli')    

    # # BUILD mednli
    mednli_train_fout = os.path.join(mt_dnn_root, 'mednli_train.json')
    mednli_dev_fout = os.path.join(mt_dnn_root, 'mednli_dev.json')
    mednli_test_fout = os.path.join(mt_dnn_root, 'mednli_test.json')
    build_data(mednli_train_data, mednli_train_fout)
    build_data(mednli_dev_data, mednli_dev_fout)
    build_data(mednli_test_data, mednli_test_fout)
    logger.info('done with mednli')

    # BUILD rqe
    rqe_train_fout = os.path.join(mt_dnn_root, 'rqe_train.json')
    rqe_dev_fout = os.path.join(mt_dnn_root, 'rqe_dev.json')
    rqe_test_fout = os.path.join(mt_dnn_root, 'rqe_test.json')
    build_data(rqe_train_data, rqe_train_fout)
    build_data(rqe_dev_data, rqe_dev_fout)
    build_data(rqe_test_data, rqe_test_fout)
    logger.info('done with rqe')

    rqe_shuff_train_fout = os.path.join(mt_dnn_root, 'rqe_shuff_train.json')
    rqe_shuff_dev_fout = os.path.join(mt_dnn_root, 'rqe_shuff_dev.json')
    rqe_shuff_test_fout = os.path.join(mt_dnn_root, 'rqe_shuff_test.json')
    build_data(rqe_shuff_train_data, rqe_shuff_train_fout)
    build_data(rqe_shuff_dev_data, rqe_shuff_dev_fout)
    build_data(rqe_shuff_test_data, rqe_shuff_test_fout)
    logger.info('done with rqe_shuff')    

    extra_features=['score','rank']
    mediqa_train_fout = os.path.join(mt_dnn_root, 'mediqa_train.json')
    mediqa_dev_fout = os.path.join(mt_dnn_root, 'mediqa_dev.json')
    mediqa_test_fout = os.path.join(mt_dnn_root, 'mediqa_test.json')
    build_data(mediqa_train_data, mediqa_train_fout, extra_features=extra_features)
    build_data(mediqa_dev_data, mediqa_dev_fout,extra_features=extra_features)
    build_data(mediqa_test_data, mediqa_test_fout,extra_features=extra_features)
    for sidx, (train_data,dev_data) in enumerate(mediqa_split_data):
        mediqa_train_fout = os.path.join(mt_dnn_root, 'mediqa_{}_train.json'.format(sidx))
        mediqa_dev_fout = os.path.join(mt_dnn_root, 'mediqa_{}_dev.json'.format(sidx))
        build_data(train_data, mediqa_train_fout, extra_features=extra_features)
        build_data(dev_data, mediqa_dev_fout, extra_features=extra_features)
    logger.info('done with mediqa')


    medquad_train_fout = os.path.join(mt_dnn_root, 'medquad_train.json')
    medquad_dev_fout = os.path.join(mt_dnn_root, 'medquad_dev.json')
    build_data(medquad_train_data, medquad_train_fout)
    build_data(medquad_dev_data, medquad_dev_fout)
    logger.info('done with medquad')    


if __name__ == '__main__':
    args = parse_args()
    if args.sci_vocab:
        # default to uncased
        bert_tokenizer = BertTokenizer.from_pretrained('../bert_models/scibert_scivocab_uncased/vocab.txt')
    elif args.cased:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    main(args)
