#!/bin/sh
#SBATCH --job-name=tetraencoder_all_datasets
#SBATCH --hint=nomultithread
#SBATCH -A ajs@gpu
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --qos=qos_gpu-dev

export TRANSFORMERS_CACHE=$ajs_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$ajs_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$ajs_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$ajs_ALL_CCFRWORK/metrics
export TORCH_HOME=$ajs_ALL_CCFRWORK/torch_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_API_KEY=`cat $WORK/wandb_key`
export WANDB_MODE=offline

source $WORK/virtualenvs/tetrenv/bin/activate
# Launch !
srun $WORK/tetraencoder/scripts/accelerate_launcher.sh \
--multi_gpu \
--fp16 \
--num_processes 8 \
--num_machines 2  \
$WORK/tetraencoder/tetraencoder/train.py \
--model_name $WORK/tetraencoder/all-mpnet-base-v2 \
--kelm_file $WORK/tetraencoder/datasets/KELM/clean_kelm.jsonl \
--tekgen_file $WORK/tetraencoder/datasets/TEKGEN/processed-tekgen-train.jsonl \
--trex_file $WORK/tetraencoder/datasets/TREx/trex_graphs.jsonl \
--eval_sq_file $WORK/tetraencoder/datasets/SQ/wd/all_splits.csv \
--eval_webnlg_wikidata_file $WORK/tetraencoder/datasets/WebNLG_Wikidata/processed_webnlg_wikidata.jsonl \
--train_batch_size 16 \
--output_dir $WORK/tetraencoder/outputs \
--num_epochs 1 \
--eval_steps 1000 \
--max_seq_length 384 \
--wandb \
--run_name all_datasets_bs2048 \
--find_unused_parameters
wait
