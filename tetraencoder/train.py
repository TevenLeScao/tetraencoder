import argparse
import os.path
import random
from datetime import datetime
from time import time

import numpy as np
import torch
from accelerate import DistributedDataParallelKwargs, Accelerator
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import TranslationEvaluator, SequentialEvaluator
from translation_evaluator_with_recall import TranslationEvaluatorWithRecall
from torch.utils.data import DataLoader

from dataset_builders import *

dataset_builders = {"msmarco": MsMarcoDataset, "kelm": KelmDataset, "gooaq": GooAqDataset, "tekgen": TekgenDataset,
                    "trex": TRexDataset}

if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    # training args
    parser.add_argument("--train_batch_size_per_gpu", default=64, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--steps_per_epoch", default=None, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--gradient_accumulation", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--hard_negatives", action="store_true")
    # i/o args
    parser.add_argument("--output_dir", default=".", type=str)
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    # evaluation dataset args
    parser.add_argument("--eval_webnlg_wikidata_file", default=None, type=str)
    parser.add_argument("--eval_gooaq_file", default=None, type=str)
    parser.add_argument("--eval_sq_file", default=None, type=str)
    # instrumentation
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--checkpoint_save_steps", default=10000, type=int)
    # distributed training args
    # TODO: remove after testing
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--find_unused_parameters", action="store_true")
    # wrap-up
    args = parser.parse_args()
    print(args)

    if args.find_unused_parameters:
        kwargs = [DistributedDataParallelKwargs(dim=0, broadcast_buffers=True, bucket_cap_mb=25,
                                               find_unused_parameters=True, check_reduction=False,
                                               gradient_as_bucket_view=False)]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    # Build model
    model = SentenceTransformer(args.model_name, add_pooling_layer=False)
    model.max_seq_length = args.max_seq_length
    word_embedding_model = model._first_module()
    word_embedding_model.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    train_datasets = [dataset_name for dataset_name in dataset_builders if
                      vars(args)[f"{dataset_name}_file"] is not None]
    name = args.run_name if args.run_name is not None else \
        f"{args.model_name.replace('/', '-')}_{'_'.join(train_datasets)}"

    model_save_path = os.path.join(args.output_dir, f'{name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    # Build datasets
    source_data = {}
    dataloaders = {}
    eval_dataloaders = {}
    for dataset_name in train_datasets:
        start_time = time()
        # for training
        input_filepath = vars(args)[f"{dataset_name}_file"]
        print(f"adding {dataset_name} to the corpus")
        dataloaders[dataset_name] = dataset_builders[dataset_name](input_filepath)
        if args.hard_negatives:
            dataloaders[dataset_name].corruption = True
        dataloaders[dataset_name] = DataLoader(dataloaders[dataset_name], shuffle=False,
                                               batch_size=args.train_batch_size_per_gpu)
        print(f"added {dataset_name} to the corpus in {time() - start_time:.3f}s")

    # Create evaluators
    evaluators = []
    task_names = []
    if args.eval_webnlg_wikidata_file is not None:
        eval_webnlg_dataset = WebNlgWikidataDataset(args.eval_webnlg_wikidata_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_webnlg_dataset.rdfs(), eval_webnlg_dataset.sentences(), show_progress_bar=False,
                                 batch_size=args.train_batch_size_per_gpu))
        task_names.append("WebNLG")
    if args.eval_gooaq_file is not None:
        eval_gooaq_dataset = GooAqDataset(args.eval_gooaq_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_gooaq_dataset.answers(), eval_gooaq_dataset.questions(), show_progress_bar=False,
                                 batch_size=args.train_batch_size_per_gpu))
        task_names.append("GOOAQ")
    if args.eval_sq_file is not None:
        eval_sq_dataset = SQDataset(args.eval_sq_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_sq_dataset.rdfs(incomplete=False), eval_sq_dataset.questions(), show_progress_bar=False,
                                 batch_size=args.train_batch_size_per_gpu))
        task_names.append("SQ_full_triplet")
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_sq_dataset.rdfs(incomplete=True), eval_sq_dataset.questions(), show_progress_bar=False,
                                 batch_size=args.train_batch_size_per_gpu))
        task_names.append("SQ_incomplete_triplet")
    if len(evaluators) == 0:
        evaluator = None
    else:
        evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

    os.makedirs(model_save_path, exist_ok=True)
    # evaluator(model, epoch=0, steps=0, output_path=model_save_path)

    if args.wandb and accelerator.is_main_process:

        import wandb
        wandb.init(project="rdf-embeddings", entity="flukeellington", name=name)
        wandb.config = {
            "model_name": args.model_name,
            "epochs": args.num_epochs,
            "learning_rate": args.lr,
            "warmup_steps": args.warmup_steps,
            "batch_size": args.train_batch_size_per_gpu * torch.cuda.device_count(),
            "max_seq_length": args.max_seq_length,
            "train_datasets": train_datasets
        }

        def train_callback(score, epoch, steps):
            wandb.log({"train_loss": score, "training_steps": steps})

        def eval_callback(scores, epoch, steps):
            for i, task_name in enumerate(task_names):
                wandb.log({f"{task_name}_recall@1": scores[i]["recall@1_src2tgt"], "training_steps": steps})
                wandb.log({f"{task_name}_recall@10": scores[i]["recall@10_src2tgt"], "training_steps": steps})
                wandb.log({f"{task_name}_MRR@10": scores[i]["mrr@10_src2tgt"], "training_steps": steps})

    else:
        train_callback = None
        eval_callback = None

    # Train the model
    model.fit(train_objectives=[(dataloader, train_loss) for dataloader in dataloaders.values()],
              evaluator=evaluator,
              evaluation_steps=args.eval_steps,
              epochs=args.num_epochs,
              steps_per_epoch=args.steps_per_epoch,
              warmup_steps=args.warmup_steps,
              use_amp=False,
              checkpoint_path=model_save_path,
              output_path=model_save_path,
              checkpoint_save_steps=args.checkpoint_save_steps,
              optimizer_params={'lr': args.lr},
              gradient_accumulation=args.gradient_accumulation,
              accelerator=accelerator,
              logging_steps=args.logging_steps,
              train_callback=train_callback,
              eval_callback=eval_callback,
              full_scores_callbacks=True
              )

    # Save the model
    model.save(model_save_path)
