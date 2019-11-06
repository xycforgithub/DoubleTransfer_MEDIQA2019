# DoubleTransfer at MEDIQA 2019: Multi-Source Transfer Learning for Natural Language Understanding in the Medical Domain

This PyTorch package implements DoubleTransfer for MEDIQA 2019 competition, as described in:

Yichong Xu, Xiaodong Liu, Chunyuan Li, Hoifung Poon and Jianfeng Gao<br/>
The BioNLP workshop, ACL 2019.<br/>
[arXiv version](https://arxiv.org/abs/1906.04382)

Please cite the above paper if you use this code. 

## Results
We report single-model results produced by this package as follows.

| Task | Score(%) | Rank |
| ------- | ------- | ------- | 
| Question Answering (QA) | **78.0 (Accuracy), 81.91 (Precision)** | **1st**
| Medical Natural Language Inference (MedNLI) | 93.8 | 3rd |
| Recognizing Question Entailment (RQE) | 66.2 | 7th |

## Quickstart 

#### Use  docker:
1. pull docker: 
   ```> docker pull yichongx/mt-dnn-mediqa```

2. run docker </br>
   ```> docker run -it --rm --runtime nvidia yichongx/mt-dnn-mediqa bash``` </br>
    Please refer to the following link if you first use docker: https://docs.docker.com/


#### Train a DoubleTransfer Model
1. preprocess data with BERT and SciBERT vocabularies
   > ./prepro.sh
2. train a model: See example codes in [run.sh](https://github.com/xycforgithub/MultiTask-MRC/blob/master/run.sh)

## Notes and Acknowledgments
The code is developed based on the original SAN code: https://github.com/kevinduh/san_mrc

Related: <a href="https://arxiv.org/abs/1901.11504">MT-DNN</a>

by
yichongx@cs.cmu.edu




