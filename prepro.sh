mkdir ../model_data/
mkdir ../mediqa_processed/
# preprocess the ground truth files for evaluation
python get_mediqa_gt_processed.py

# preprocessing, uncased
python prepro_mediqa.py
# preprocessing, cased
python prepro_mediqa.py --cased
# preprocessing, uncased, with SciBERT vocabulary
python prepro_mediqa.py --sci_vocab

