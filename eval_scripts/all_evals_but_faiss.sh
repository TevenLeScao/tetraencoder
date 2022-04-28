accelerate launch tetraencoder/eval_along_training.py \
--checkpoints_folder $1 \
--eval_batch_size_per_gpu 64 \
--eval_simple_mpww_file /home/teven_huggingface_co/tetraencoder/datasets/MPWW/translation_style_mpww.jsonl \
--eval_webnlg_wikidata_file datasets/WebNLG_Wikidata/processed_webnlg_wikidata.jsonl \
--eval_webnlg_dbpedia_file datasets/WebNLG_DBpedia/processed_webnlg_dbpedia.jsonl \
--max_seq_length 384 \
--wandb \
--accelerate \
--run_name $2
