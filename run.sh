
python train.py --train_datasets mednli,rqe,mediqa,medquad --save_last --save_best --mediqa_score adjusted --mediqa_score_offset -2.0 --freeze_bert_first --batch_size 16 --max_seq_len 384 --data_dir ../data/mediqa_processed/mt_dnn_mediqa_384_v2/ --init_checkpoint /path/to/pretrained/model/ --float_medquad --external_datasets mnli --mtl_opt 0 --output_dir /output/path 
