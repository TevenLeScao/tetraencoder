python /home/teven_huggingface_co/tetraencoder/tetraencoder/eval_along_training.py \
--checkpoints_folder $1 \
--only_last_checkpoint \
--eval_batch_size_per_gpu 256 \
--max_seq_length 384 \
--run_name baseline \
--eval_mpww_file /home/teven_huggingface_co/tetraencoder/datasets/MPWW/mpww.jsonl \
--eval_mpww_passages_file /home/teven_huggingface_co/tetraencoder/datasets/MPWW/passages_with_matches.csv \
--faiss_index_training_samples 163840 \
--run_name $2 \
--wandb