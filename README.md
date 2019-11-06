# DoubleTransfer at MEDIQA 2019: </br> Multi-Source Transfer Learning for Natural Language Understanding in the Medical Domain

This PyTorch package implements DoubleTransfer for [the MEDIQA 2019 competition](https://www.aclweb.org/anthology/W19-5039.pdf), as described in:

Yichong Xu, Xiaodong Liu, Chunyuan Li, Hoifung Poon and Jianfeng Gao<br/>
DoubleTransfer at MEDIQA 2019: Multi-Source Transfer Learning for Natural Language Understanding in the Medical Domain </br> 
The BioNLP workshop, ACL 2019.<br/>
[arXiv version](https://arxiv.org/abs/1906.04382)

Please cite the above paper if you use this code. 

## Results
We report results produced by this package as follows.

|                     Task                    |                Score(%)                |   Rank  |
|:-------------------------------------------:|:--------------------------------------:|:-------:|
|           Question Answering (QA)           | **78.0 (Accuracy), 81.91 (Precision)** | **1st** |
| Medical Natural Language Inference (MedNLI) |                  93.8                  |   3rd   |
|    Recognizing Question Entailment (RQE)    |                  66.2                  |   7th   |

## Quickstart 

#### Use  docker:
1. pull docker: </br>
   ```> docker pull yichongx/doubletransfer_mediqa2019```

2. run docker </br>
   ```> docker run -it --rm --runtime nvidia yichongx/doubletransfer_mediqa2019 bash``` </br>
    Please refer to the following link if you first use docker: https://docs.docker.com/


#### Train a DoubleTransfer Model
1. Download the data using links in [the MEDIQA 2019 website](https://sites.google.com/view/mediqa2019).
2. Prepare MNLI data as well as pretrained BERT models.
   ``` > ./download.sh```
1. preprocess data with BERT and SciBERT vocabularies</br>
   ``` > ./prepro.sh```
2. train a model using train.py. </br>
   ``` > python train.py --train_datasets mednli,rqe,mediqa,medquad --save_last --save_best --mediqa_score adjusted --mediqa_score_offset -2.0 --freeze_bert_first --batch_size 16 --max_seq_len 384 --data_dir ../data/mediqa_processed/mt_dnn_mediqa_384_v2/ --init_checkpoint /path/to/pretrained/model/ --float_medquad --external_datasets mnli --mtl_opt 0 --output_dir /output/path ```
   See example codes in [run.sh](https://github.com/xycforgithub/DoubleTransfer_MEDIQA2019/blob/master/run.sh)
3. To ensemble predictions:</br>
   ``` > python ensemble_preds.py /path/to/file1/ /path/to/file2/ ```</br>
   All the input files will be ensembled.

## Notes and Acknowledgments
The code is developed based on the original MT-DNN code: https://github.com/namisan/mt-dnn

Related: <a href="https://arxiv.org/abs/1809.06963">MultiTask-MRC</a>
<a href="https://arxiv.org/abs/1901.11504">MT-DNN</a>


by
yichongx@cs.cmu.edu




