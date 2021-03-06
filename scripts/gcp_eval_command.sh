accelerate launch tetraencoder/train.py  \
--model_name_or_path sentence-transformers/all-mpnet-base-v2 \
--kelm_file datasets/KELM/clean_kelm.jsonl \
--tekgen_file datasets/TEKGEN/processed-tekgen-train.jsonl \
--trex_file datasets/TREx/trex_graphs.jsonl \
--eval_webnlg_wikidata_file datasets/WebNLG_Wikidata/processed_webnlg_wikidata.jsonl \
--train_batch_size 24 \
--eval_batch_size 80 \
--output_dir outputs \
--num_epochs -1 \
--eval_steps 2000 \
--checkpoint_save_steps 2000 \
--max_seq_length 384 \
--find_unused_parameters \
--replaced_negatives \
--wandb \
--run_name baseline
