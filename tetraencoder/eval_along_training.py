import argparse
import glob
import os.path
import random
from operator import xor

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from accelerate import Accelerator
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from dataset_wrappers import *
from train import build_evaluators

datasets.logging.set_verbosity_error()


def evaluate(model_path, sequential_evaluator, task_names, batch_size, wandb_log=False, max_seq_length=512, fp16=False,
             is_main_process=True):
    # Build model
    model = SentenceTransformer(model_path, add_pooling_layer=False)
    model.max_seq_length = max_seq_length
    if fp16:
        model = model.half()

    # reloading an existing model
    try:
        training_step = int(model_path.split("/")[-1])
    except ValueError:
        training_step = -1
    seen_items = batch_size * training_step
    eval_path = os.path.join(model_path, "eval_out_of_training")
    os.makedirs(eval_path, exist_ok=True)

    # evaluating
    for evaluator in sequential_evaluator.evaluators:
        evaluator.show_progress_bar = True
    main_score, scores = sequential_evaluator(model, output_path=eval_path, steps=seen_items, epoch=1,
                                              return_all_scores=True)

    # logging
    if wandb_log and is_main_process:
        for i, task_name in enumerate(task_names):
            if task_name in ["MPWW", "MPWW_filtered"]:
                wandb.log({f"{task_name}_accuracy@1": scores[i]["accuracy@k"][1],
                           "data_points": seen_items})
                wandb.log({f"{task_name}_recall@10": scores[i]["recall@k"][10],
                           "data_points": seen_items})
                wandb.log(
                    {f"{task_name}_precision@10": scores[i]["precision@k"][10],
                     "data_points": seen_items})
                wandb.log(
                    {f"{task_name}_MRR@10": scores[i]["mrr@k"][10], "data_points": seen_items})
            else:
                wandb.log({f"{task_name}_recall@1": scores[i]["recall@1_src2tgt"], "data_points": seen_items})
                wandb.log(
                    {f"{task_name}_recall@10": scores[i]["recall@10_src2tgt"], "data_points": seen_items})
                wandb.log({f"{task_name}_MRR@10": scores[i]["mrr@10_src2tgt"], "data_points": seen_items})


def estimate_base_model(name):
    for model_name in ["all-mpnet-base-v2", "all_bs160_allneg", "all_bs320_vanilla", "all_bs192_hardneg"]:
        if model_name in name:
            return model_name
    return None


def estimate_biencoder(name):
    if "cross_" in name:
        return False
    elif "bi_" in name:
        return True
    return None


if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--fp16", action="store_true")
    # i/o args
    parser.add_argument("--model_name_or_path", default=None, required=False, type=str)
    parser.add_argument("--checkpoints_folder", default=None, required=False, type=str)
    parser.add_argument("--only_last_checkpoint", action="store_true")
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    # various input sizes
    parser.add_argument("--eval_corpus_chunk_size", default=16384, type=int)
    parser.add_argument("--eval_batch_size_per_gpu", default=64, type=int)
    parser.add_argument("--index_training_samples", default=4096, type=int)
    parser.add_argument("--faiss_gpu", action="store_true")
    # evaluation dataset args
    parser.add_argument("--eval_webnlg_wikidata_file", default=None, type=str)
    parser.add_argument("--eval_webnlg_dbpedia_file", default=None, type=str)
    parser.add_argument("--eval_gooaq_file", default=None, type=str)
    parser.add_argument("--eval_sq_file", default=None, type=str)
    parser.add_argument("--eval_simple_mpww_file", default=None, type=str)
    parser.add_argument("--eval_mpww_file", default=None, type=str)
    parser.add_argument("--eval_mpww_passages_file", default=None, type=str)
    # instrumentation
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", default="test", type=str)
    # use accelerate
    parser.add_argument("--accelerate", action="store_true")
    # wrap-up
    args = parser.parse_args()
    print(args)

    if args.accelerate:
        accelerator = Accelerator()
        is_main_process = accelerator.is_main_process
    else:
        accelerator = None
        is_main_process = True
    sequential_evaluator, task_names = build_evaluators(args)

    try:
        train_datasets, batch_size, neg_type = args.run_name.split("_")
        train_datasets = ["kelm", "tekgen", "trex"] if train_datasets == "all" else [train_datasets]
        batch_size = int(batch_size[2:])
        neg_type = neg_type[:-3]
    except ValueError:
        train_datasets = []
        batch_size = 0
        neg_type = None

    if args.checkpoints_folder is not None:
        assert os.path.isdir(args.checkpoints_folder)
        assert args.model_name_or_path is None, "Can't pass both model_name_or_path and checkpoints_folder"
        paths = glob.glob(args.checkpoints_folder + "/[0-9]*")
        paths = sorted(paths, key=lambda folder: int(folder.split("/")[-1]))
        if args.only_last_checkpoint:
            paths = paths[-1:]
    else:
        assert args.model_name_or_path is not None, "Must pass both model_name_or_path or checkpoints_folder"
        paths = [args.model_name_or_path]

    if args.wandb and is_main_process:
        import wandb

        wandb.init(project="rdf-embeddings-evals-paper", entity="flukeellington", name=args.run_name,
                   config={
                       "batch_size": batch_size,
                       "train_datasets": train_datasets,
                       "negatives": neg_type,
                       "max_seq_length": args.max_seq_length,
                       "finetuned": "finetuned" in paths[0],
                       "base_model": estimate_base_model(paths[0]),
                       "biencoder": estimate_biencoder(paths[0])
                   })

    for path in tqdm(paths):
        evaluate(path, sequential_evaluator, task_names, batch_size, wandb_log=args.wandb,
                 max_seq_length=args.max_seq_length, is_main_process=is_main_process, fp16=args.fp16)
