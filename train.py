# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
from shutil import copyfile, move

import numpy as np
import torch

from data_utils.mediqa_utils import submit, eval_model,mediqa_name_list
from data_utils.label_map import DATA_META, GLOBAL_MAP, DATA_TYPE, DATA_SWAP, TASK_TYPE, generate_decoder_opt
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from data_utils.mediqa2019_evaluator_allTasks_final import eval_mediqa_official
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel
from bert.modeling import BertModel
from bert.modeling import BertConfig
import pdb

def model_config(parser):
    parser.add_argument('--update_bert_opt',  default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_false')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='3,3,2')
    parser.add_argument('--mtl_opt', type=int, default=1)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    return parser

def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file(under model_dir).')
    parser.add_argument("--init_checkpoint", default='../data/bert_data/bert_model_base.pt', type=str)
    parser.add_argument('--init_config', default=None, type=str,
     help='if specified, will load a huggingface-style checkpoint with a separate config file.')
    parser.add_argument('--finetune',action='store_true', help='will only resume parameters and delete the optimizer states, etc.')
    parser.add_argument('--resume_scoring', type=int, default=None, help='finetune from this scoring file.')
    parser.add_argument('--retain_scoring',action='store_true',help='retain all previous scoring units.')
    parser.add_argument('--data_dir', default='../data/mt_dnn_mediqa')
    parser.add_argument('--data_root',default='../data/', type=str, help='contains raw data folders.')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--train_datasets', default='mednli,rqe')
    parser.add_argument('--external_datasets',default='')
    parser.add_argument('--external_include_ratio', default=1.0, type=float, help='include ratio*in-domain data samples for external data.')
    parser.add_argument('--test_datasets', default=None)
    parser.add_argument('--predict_split', default=None, type=str,
        help='if specified, will skip training and just predict the specified split.')
    parser.add_argument('--mediqa_score', default='adjusted', type=str, 
        help='mediqa target score. Either adjusted or raw.')
    parser.add_argument('--mediqa_pairloss',default=None, type=str,
        help='Pairwise loss for mediqa (default None, either hinge or logistic). must have an even batch size.')
    parser.add_argument('--mediqa_score_offset',default=0.0, type=float,
        help='offset the mediqa target score to make learning easier.')
    parser.add_argument('--mediqa_eval_more', action='store_true', help='evaluate more metrices for mediqa.')
    parser.add_argument('--float_medquad', action='store_true',
        help='set medquad to be a regression task to help mediqa.')
    parser.add_argument('--answer_relu', action='store_true', 
        help='add a relu after linear of finetune layers for regression tasks.')
    parser.add_argument('--hinge_lambda', default=1.0, type=float,
        help='lambda for hinge loss.')
    parser.add_argument('--mtl_observe_datasets',default='mednli,rqe',type=str,
        help='use these dataset to keep best model.')
    return parser

def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_eval', type=int, default=None)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)


    # EMA
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--ema_gamma', type=float, default=0.995)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--freeze_bert_first', action='store_true',help='fix bert layers at the first epoch.')
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='../model_data/mt-dnn-mediqa/')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--task_config_path', type=str, default='configs/tasks_config.json')
    parser.add_argument('--not_save_model', action='store_true', help='not save model.')
    parser.add_argument('--test_mode', action='store_true', help='skip training.')
    parser.add_argument('--save_last',action='store_true', help='only save last epoch.')
    parser.add_argument('--save_best', action='store_true', help='save the best mtl-perf model.')

    return parser

parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
args = parser.parse_args()

args.mtl_observe_datasets = args.mtl_observe_datasets.split(',')
if args.float_medquad:
    TASK_TYPE['medquad'] = 1
    DATA_META['medquad'] = 1

args.float_target = False

if args.batch_size_eval is None:
    args.batch_size_eval=args.batch_size

output_dir = args.output_dir
data_dir = args.data_dir
if args.test_datasets is None:
    args.test_datasets = args.train_datasets
args.train_datasets = args.train_datasets.split(',')
args.test_datasets = args.test_datasets.split(',')
if len(args.train_datasets)==1:
    args.mtl_observe_datasets = args.train_datasets
args.external_datasets = args.external_datasets.split(',') if args.external_datasets!='' else []
args.train_datasets = args.train_datasets + args.external_datasets



pprint(args)

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = os.path.join(output_dir, args.log_file)
logger =  create_logger(__name__, to_disk=True, log_file=log_path)
logger.info(args.answer_opt)

tasks_config = {}
if os.path.exists(args.task_config_path):
    with open(args.task_config_path, 'r') as reader:
        tasks_config = json.loads(reader.read())

def dump(path, data):
    with open(path ,'w') as f:
        json.dump(data, f)

def main():
    logger.info('Launching the MT-DNN training')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size
    train_data_list = []
    tasks = {}
    tasks_class = {}
    nclass_list = []
    decoder_opts = []
    dropout_list = []

    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks: continue
        assert prefix in DATA_META
        assert prefix in DATA_TYPE
        data_type = DATA_TYPE[prefix]
        nclass = DATA_META[prefix]
        task_id = len(tasks)
        if args.mtl_opt > 0:
            task_id = tasks_class[nclass] if nclass in tasks_class else len(tasks_class)

        task_type = TASK_TYPE[prefix]
        pw_task = False

        dopt = generate_decoder_opt(prefix, opt['answer_opt'])
        if task_id < len(decoder_opts):
            decoder_opts[task_id] = min(decoder_opts[task_id], dopt)
        else:
            decoder_opts.append(dopt)

        if prefix not in tasks:
            tasks[prefix] = len(tasks)
            if args.mtl_opt < 1: nclass_list.append(nclass)

        if (nclass not in tasks_class):
            tasks_class[nclass] = len(tasks_class)
            if args.mtl_opt > 0: nclass_list.append(nclass)

        dropout_p = args.dropout_p
        if tasks_config and prefix in tasks_config:
            dropout_p = tasks_config[prefix]
        dropout_list.append(dropout_p)

        train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
        logger.info('Loading {} as task {}'.format(train_path, task_id))
        train_data = BatchGen(BatchGen.load(train_path, True, pairwise=pw_task, maxlen=args.max_seq_len, 
                                        opt=opt, dataset=dataset),
                                batch_size=batch_size,
                                dropout_w=args.dropout_w,
                                gpu=args.cuda,
                                task_id=task_id,
                                maxlen=args.max_seq_len,
                                pairwise=pw_task,
                                data_type=data_type,
                                task_type=task_type,
                                dataset_name=dataset)
        train_data.reset()
        train_data_list.append(train_data)

    opt['answer_opt'] = decoder_opts
    opt['tasks_dropout_p'] = dropout_list

    args.label_size = ','.join([str(l) for l in nclass_list])
    logger.info(args.label_size)
    dev_data_list = []
    test_data_list = []
    for dataset in args.test_datasets:
        prefix = dataset.split('_')[0]
        task_id = tasks_class[DATA_META[prefix]] if args.mtl_opt > 0 else tasks[prefix]
        task_type = TASK_TYPE[prefix]

        pw_task = False

        assert prefix in DATA_TYPE
        data_type = DATA_TYPE[prefix]

        if args.predict_split is not None:
            dev_path = os.path.join(data_dir, '{}_{}.json'.format(dataset, 
                args.predict_split))
        else:
            dev_path = os.path.join(data_dir, '{}_dev.json'.format(dataset))
        dev_data = None
        if os.path.exists(dev_path):
            dev_data = BatchGen(BatchGen.load(dev_path, False, pairwise=pw_task, maxlen=args.max_seq_len,
                                            opt=opt, dataset=dataset),
                                  batch_size=args.batch_size_eval,
                                  gpu=args.cuda, is_train=False,
                                  task_id=task_id,
                                  maxlen=args.max_seq_len,
                                  pairwise=pw_task,
                                  data_type=data_type,
                                  task_type=task_type,
                                  dataset_name=dataset)
        dev_data_list.append(dev_data)

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data = BatchGen(BatchGen.load(test_path, False, pairwise=pw_task, 
                                            maxlen=args.max_seq_len,opt=opt, dataset=dataset),
                                  batch_size=args.batch_size_eval,
                                  gpu=args.cuda, is_train=False,
                                  task_id=task_id,
                                  maxlen=args.max_seq_len,
                                  pairwise=pw_task,
                                  data_type=data_type,
                                  task_type=task_type,
                                  dataset_name=dataset)
        test_data_list.append(test_data)

    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)

    all_iters =[iter(item) for item in train_data_list]
    all_lens = [len(bg) for bg in train_data_list]
    num_all_batches = args.epochs * sum(all_lens)

    if len(args.external_datasets) > 0 and args.external_include_ratio > 0:
        num_in_domain_batches = args.epochs* sum(all_lens[:-len(args.external_datasets)])
        num_all_batches = num_in_domain_batches * (1 + args.external_include_ratio)
    # pdb.set_trace()

    model_path = args.init_checkpoint
    state_dict = None

    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        if args.init_config is not None: # load huggingface model
            config = json.load(open(args.init_config))
            state_dict={'config':config, 'state':state_dict}
        if args.finetune:
            # only resume config and state
            del_keys=set(state_dict.keys())-set(['config','state'])
            for key in del_keys:
                del state_dict[key]
            resume_configs=json.load(open('config/resume_configs.json'))
            del_keys=set(state_dict['config'].keys())-set(resume_configs)
            for key in del_keys:
                del state_dict['config'][key]
            if args.resume_scoring is not None:                    
                for key in state_dict['state'].keys():
                    if 'scoring_list.0' in key:
                        state_dict['state'][key]=state_dict['state'][key.replace('0',str(args.resume_scoring))]
                        # other scorings will be deleted during loading process, since finetune only has one task
            elif not args.retain_scoring:
                del_keys = [k for k in state_dict['state'] if 'scoring_list' in k]
                for key in del_keys:                    
                    print('deleted previous weight:',key)
                    del state_dict['state'][key]
        config = state_dict['config']
        config['attention_probs_dropout_prob'] = args.bert_dropout_p
        config['hidden_dropout_prob'] = args.bert_dropout_p
        opt.update(config)
    else:
        logger.error('#' * 20)
        logger.error('Could not find the init model!\n The parameters will be initialized randomly!')
        logger.error('#' * 20)
        config = BertConfig(vocab_size_or_config_json_file=30522).to_dict()
        opt.update(config)

    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=num_all_batches)
    ####model meta str
    headline = '############# Model Arch of MT-DNN #############'
    ###print network
    # logger.info('\n{}\n{}\n'.format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
        writer.write('\n{}\n{}\n'.format(headline, model.network))

    logger.info("Total number of params: {}".format(model.total_param))

    if args.freeze_layers > 0:
        model.network.freeze_layers(args.freeze_layers)

    if args.cuda:
        model.cuda()
    best_epoch=-1
    best_performance=0 
    best_dataset_performance={dataset:{'perf':0,'epoch':-1} for dataset in args.mtl_observe_datasets}
    for epoch in range(0, args.epochs):
        logger.warning('At epoch {}'.format(epoch))
        if epoch==0 and args.freeze_bert_first:
            model.network.freeze_bert()
            logger.warning('Bert freezed.')
        if epoch==1 and args.freeze_bert_first:
            model.network.unfreeze_bert()
            logger.warning('Bert unfreezed.')
        start = datetime.now()
        all_indices=[]
        if len(args.external_datasets)> 0 and args.external_include_ratio>0:
            main_indices = []
            extra_indices = []
            for data_idx,batcher in enumerate(train_data_list):
                if batcher.dataset_name not in args.external_datasets:
                    main_indices += [data_idx] * len(batcher)
                else:
                    extra_indices += [data_idx] * len(batcher)

            random_picks=int(min(len(main_indices) * args.external_include_ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if args.mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()
        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if args.mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if args.mix_opt < 1:
            random.shuffle(all_indices)
        if args.test_mode:
            all_indices=all_indices[:2]
        if args.predict_split is not None:
            all_indices=[]
            dev_split=args.predict_split
        else:
            dev_split='dev'

        for i in range(len(all_indices)):
            task_id = all_indices[i]
            batch_meta, batch_data= next(all_iters[task_id])
            model.update(batch_meta, batch_data)
            if (model.updates) % args.log_per_updates == 0 or model.updates == 1:
                logger.info('Task [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(task_id,
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(all_indices) - i - 1)).split('.')[0]))
        os.system('nvidia-smi')
        for train_data in train_data_list:
            train_data.reset()        

        this_performance={}

        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            dev_data = dev_data_list[idx]
            if dev_data is not None:
                dev_metrics, dev_predictions, scores, golds, dev_ids= eval_model(model, dev_data, dataset=prefix,
                                                                                 use_cuda=args.cuda)
                score_file = os.path.join(output_dir, '{}_{}_scores_{}.json'.format(dataset, dev_split, epoch))
                results = {'metrics': dev_metrics, 'predictions': dev_predictions, 'uids': dev_ids, 'scores': scores}
                dump(score_file, results)
                official_score_file = os.path.join(output_dir, '{}_{}_scores_{}.csv'.format(dataset, dev_split, epoch))
                submit(official_score_file, results,dataset_name=prefix, threshold=2.0+args.mediqa_score_offset)
                if prefix in mediqa_name_list:
                    logger.warning('self test numbers:{}'.format(dev_metrics))
                    if '_' in dataset:
                        affix = dataset.split('_')[1]
                        ground_truth_path=os.path.join(args.data_root,'mediqa/task3_qa/gt_{}_{}.csv'.format(dev_split,affix))
                    else:
                        ground_truth_path=os.path.join(args.data_root,'mediqa/task3_qa/gt_{}.csv'.format(dev_split))
                    official_result=eval_mediqa_official(pred_path=official_score_file, ground_truth_path=ground_truth_path, 
                        eval_qa_more=args.mediqa_eval_more)
                    logger.warning("MediQA dev eval result:{}".format(official_result))
                    if args.mediqa_eval_more:
                        dev_metrics={'ACC':official_result['score']*100,'Spearman':official_result['score_secondary']*100,
                                    'F1':dev_metrics['F1'], 'MRR':official_result['meta']['MRR'], 'MAP':official_result['MAP'],
                                    'P@1':official_result['meta']['P@1']}
                    else:
                        dev_metrics={'ACC':official_result['score']*100,'Spearman':official_result['score_secondary']*100}

                for key, val in dev_metrics.items():
                    logger.warning("Task {0} -- epoch {1} -- Dev {2}: {3:.3f}".format(dataset, epoch, key, val))
            if args.predict_split is not None:
                continue
            print('args.mtl_observe_datasets:',args.mtl_observe_datasets, dataset)
            if dataset in args.mtl_observe_datasets:
                this_performance[dataset]=np.mean([val for val in dev_metrics.values()])
            test_data = test_data_list[idx]
            if test_data is not None:
                test_metrics, test_predictions, scores, golds, test_ids= eval_model(model, test_data, dataset=prefix,
                                                                                 use_cuda=args.cuda, with_label=False)
                for key, val in test_metrics.items():
                    logger.warning("Task {0} -- epoch {1} -- Test {2}: {3:.3f}".format(dataset, epoch, key, val))
                score_file = os.path.join(output_dir, '{}_test_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
                dump(score_file, results)
                # if dataset in mediqa_name_list:
                official_score_file = os.path.join(output_dir, '{}_test_scores_{}.csv'.format(dataset, epoch))
                submit(official_score_file, results,dataset_name=prefix, threshold=2.0+args.mediqa_score_offset)
                logger.info('[new test scores saved.]')
        print('this_performance:',this_performance)
        if args.predict_split is not None:
            break
        epoch_performance = sum([val for val in this_performance.values()])
        if epoch_performance>best_performance:
            print('changed:',epoch_performance,best_performance)
            best_performance=epoch_performance
            best_epoch=epoch

        for dataset in args.mtl_observe_datasets:
            if best_dataset_performance[dataset]['perf']<this_performance[dataset]:
                best_dataset_performance[dataset]={'perf':this_performance[dataset],
                                                   'epoch':epoch} 


        print('current best:',best_performance,'at epoch', best_epoch)
        if not args.not_save_model:
            model_name = 'model_last.pt' if args.save_last else 'model_{}.pt'.format(epoch) 
            model_file = os.path.join(output_dir, model_name)
            if args.save_last and os.path.exists(model_file):
                model_temp=os.path.join(output_dir, 'model_secondlast.pt')
                copyfile(model_file, model_temp)
            model.save(model_file)
            if args.save_best and best_epoch==epoch:
                best_path = os.path.join(output_dir,'best_model.pt')
                copyfile(model_file,best_path)
                for dataset in args.mtl_observe_datasets:
                    if best_dataset_performance[dataset]['epoch']==epoch:
                        best_path = os.path.join(output_dir,'best_model_{}.pt'.format(dataset))
                        copyfile(model_file,best_path)


if __name__ == '__main__':
    main()
