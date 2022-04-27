python tetraencoder/eval_along_training.py \
--checkpoints_folder $1 \
--eval_batch_size_per_gpu 64 \
--eval_simple_mpww_file /home/teven_huggingface_co/tetraencoder/datasets/MPWW/translation_style_mpww.jsonl \
--max_seq_length 384 \
--wandb \
--run_name $2
