accelerate launch tetraencoder/eval_along_training.py \
--checkpoints_folder $1 \
--eval_batch_size_per_gpu 256 \
--eval_sq_file datasets/SQ/wd/all_splits.csv \
--eval_webnlg_wikidata_file datasets/WebNLG_Wikidata/processed_webnlg_wikidata.jsonl \
--eval_webnlg_dbpedia_file datasets/WebNLG_DBpedia/processed_webnlg_dbpedia.jsonl \
--max_seq_length 384 \
--wandb \
--run_name $2
