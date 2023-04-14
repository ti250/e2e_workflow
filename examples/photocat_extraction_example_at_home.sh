export BERT_MODEL_NAME="deepset/deberta-v3-large-squad2"
# export DOCUMENT_DIR="datasets/processed/photocat_train"
export DOCUMENT_DIR="pankaj_sample_articles/article_sample"
export OUTPUT_DIR="photocat_datasets/cde_run/pankaj_article_sample-2"
export USE_MPI=0

python e2e_workflow/examples/photocat_extraction.py
