import argparse
import glob
import os.path
import random

import numpy as np
import torch
from accelerate import Accelerator
from datasets import tqdm
from sentence_transformers import SentenceTransformer

from dataset_builders import *
from train import build_evaluators

dataset_builders = {"msmarco": MsMarcoDataset, "kelm": KelmDataset, "gooaq": GooAqDataset, "tekgen": TekgenDataset,
                    "trex": TRexDataset}


def evaluate(model_path, sequential_evaluator, task_names, batch_size, wandb_log=False, max_seq_length=512,
             is_main_process=True):
    # Build model
    model = SentenceTransformer(model_path, add_pooling_layer=False)
    model.max_seq_length = max_seq_length

    # reloading an existing model
    training_step = int(model_path.split("/")[-1])
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
            if task_name == "MPWW":
                wandb.log({f"{task_name}_accuracy@1": scores[i]["cos_sim"]["accuracy@k"][1],
                           "data_points": seen_items})
                wandb.log({f"{task_name}_recall@10": scores[i]["cos_sim"]["recall@k"][10],
                           "data_points": seen_items})
                wandb.log(
                    {f"{task_name}_precision@10": scores[i]["cos_sim"]["precision@k"][10],
                     "data_points": seen_items})
                wandb.log(
                    {f"{task_name}_MRR@10": scores[i]["cos_sim"]["mrr@k"][10], "data_points": seen_items})
            else:
                wandb.log({f"{task_name}_recall@1": scores[i]["recall@1_src2tgt"], "data_points": seen_items})
                wandb.log(
                    {f"{task_name}_recall@10": scores[i]["recall@10_src2tgt"], "data_points": seen_items})
                wandb.log({f"{task_name}_MRR@10": scores[i]["mrr@10_src2tgt"], "data_points": seen_items})


if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--checkpoints_folder", default=None, required=True, type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    # i/o args
    parser.add_argument("--output_dir", default=".", type=str)
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    # evaluation dataset args
    parser.add_argument("--eval_batch_size_per_gpu", default=64, type=int)
    parser.add_argument("--eval_webnlg_wikidata_file", default=None, type=str)
    parser.add_argument("--eval_webnlg_dbpedia_file", default=None, type=str)
    parser.add_argument("--eval_gooaq_file", default=None, type=str)
    parser.add_argument("--eval_sq_file", default=None, type=str)
    parser.add_argument("--eval_mpww_file", default=None, type=str)
    parser.add_argument("--eval_mpww_passages_file", default=None, type=str)
    parser.add_argument("--eval_corpus_chunk_size", default=16384, type=int)
    # instrumentation
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", default="test", type=str)
    # wrap-up
    args = parser.parse_args()
    print(args)
    assert os.path.isdir(args.checkpoints_folder)

    accelerator = Accelerator()
    sequential_evaluator, task_names = build_evaluators(args)

    train_datasets, batch_size, neg_type = args.run_name.split("_")
    train_datasets = ["kelm", "tekgen", "trex"] if train_datasets == "all" else [train_datasets]
    batch_size = int(batch_size[2:])
    neg_type = neg_type[:-3]

    if args.wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project="rdf-embeddings-evals", entity="flukeellington", name=args.run_name)
        wandb.config = {
            "model_name_or_path": args.model_name_or_path,
            "batch_size": batch_size,
            "train_datasets": train_datasets,
            "negatives": neg_type,
            "max_seq_length": args.max_seq_length,
        }

    for subfolder in tqdm(glob.glob(args.checkpoints_folder + "/[0-9]*")):
        evaluate(subfolder, sequential_evaluator, task_names, batch_size, wandb_log=args.wandb,
                 max_seq_length=args.max_seq_length, is_main_process=accelerator.is_main_process)
