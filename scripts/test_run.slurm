#!/bin/sh
#SBATCH --job-name=test
#SBATCH --hint=nomultithread
#SBATCH -A ajs@gpu
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --nodes=8
#SBATCH --qos=qos_gpu-dev
#SBATCH -o /gpfsscratch/rech/ajs/uhk85as/tetraencoder_xps/log_test-%j.out
#SBATCH -e /gpfsscratch/rech/ajs/uhk85as/tetraencoder_xps/log_test-%j.err

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
--num_processes 32 \
--num_machines 8  \
$WORK/tetraencoder/tetraencoder/train.py \
--model_name_or_path $WORK/tetraencoder/all-mpnet-base-v2 \
--kelm_file $WORK/tetraencoder/outputs/dataset_embeddings/from_all/kelm.jsonl \
--tekgen_file $WORK/tetraencoder/outputs/dataset_embeddings/from_all/tekgen.jsonl \
--trex_file $WORK/tetraencoder/outputs/dataset_embeddings/from_all/trex.jsonl \
--similarity_fraction_to_keep 0.75 \
--train_batch_size 24 \
--replaced_negatives \
--output_dir $SCRATCH/tetraencoder_xps \
--num_epochs 1 \
--eval_steps 1000 \
--checkpoint_save_steps 1000 \
--max_seq_length 384 \
--run_name test \
--find_unused_parameters
wait
