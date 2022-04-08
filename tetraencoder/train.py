import argparse
import os.path
import random
from datetime import datetime
from time import time

from datasets import load_dataset
import numpy as np
import torch
from accelerate import DistributedDataParallelKwargs, Accelerator
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import TranslationEvaluator, SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim

from faiss_ir_evaluator import FaissIREvaluator
from translation_evaluator_with_recall import TranslationEvaluatorWithRecall
from torch.utils.data import DataLoader

from dataset_builders import *

dataset_builders = {"msmarco": MsMarcoDataset, "kelm": KelmDataset, "gooaq": GooAqDataset, "tekgen": TekgenDataset,
                    "trex": TRexDataset}


def build_evaluators(args):    # Create evaluators
    evaluators = []
    task_names = []
    if args.eval_webnlg_wikidata_file is not None:
        print("adding WebNLG eval data")
        eval_webnlg_dataset = WebNlgDataset(args.eval_webnlg_wikidata_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_webnlg_dataset.rdfs(), eval_webnlg_dataset.sentences(),
                                           show_progress_bar=False,
                                           batch_size=args.eval_batch_size_per_gpu))
        task_names.append("WebNLG")

    if args.eval_webnlg_dbpedia_file is not None:
        print("adding WebNLG-DB eval data")
        eval_webnlg_dataset = WebNlgDataset(args.eval_webnlg_dbpedia_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_webnlg_dataset.rdfs(), eval_webnlg_dataset.sentences(),
                                           show_progress_bar=False,
                                           batch_size=args.eval_batch_size_per_gpu))
        task_names.append("WebNLG-DB")

    if args.eval_gooaq_file is not None:
        print("adding GOOAQ eval data")
        eval_gooaq_dataset = GooAqDataset(args.eval_gooaq_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_gooaq_dataset.answers(), eval_gooaq_dataset.questions(),
                                           show_progress_bar=False,
                                           batch_size=args.eval_batch_size_per_gpu))
        task_names.append("GOOAQ")

    if args.eval_sq_file is not None:
        print("adding SQ eval data")
        eval_sq_dataset = SQDataset(args.eval_sq_file)
        evaluators.append(
            TranslationEvaluatorWithRecall(eval_sq_dataset.rdfs(incomplete=False), eval_sq_dataset.questions(),
                                           show_progress_bar=False,
                                           batch_size=args.eval_batch_size_per_gpu))
        task_names.append("SQ_full_triplet")

    if args.eval_mpww_file is not None:
        print("adding MPWW queries eval data")
        mpww = MPWWDataset(args.eval_mpww_file)
        if args.eval_mpww_passages_file is not None:
            queries = {i: query for i, query in enumerate(mpww.rdfs())}
            print("adding MPWW passages corpus eval data")
            print("adding MPWW passages corpus eval data")
            passages = load_dataset("csv", data_files=args.eval_mpww_passages_file)["train"]
            corpus = {i: passage for i, passage in enumerate(passages["text"])}
            relevant_docs = {match: {i} for i, match in enumerate(passages["mpww_match"]) if match is not None}
            evaluators.append(
                FaissIREvaluator(queries, corpus, relevant_docs, show_progress_bar=False,
                                 corpus_chunk_size=args.eval_corpus_chunk_size, precision_recall_at_k=[10],
                                 accuracy_at_k=[1], batch_size=args.eval_batch_size_per_gpu,
                                 score_function='cos_sim', index_training_samples=4096))
            task_names.append("MPWW_fullIR")
        else:
            evaluators.append(
                TranslationEvaluatorWithRecall(mpww.rdfs(), mpww.sentences(),
                                               show_progress_bar=False,
                                               batch_size=args.eval_batch_size_per_gpu))
            task_names.append("MPWW_partial")
    else:
        assert args.eval_mpww_file is None

    if len(evaluators) == 0:
        sequential_evaluator = None
    else:
        sequential_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

    return sequential_evaluator, task_names


if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    # training args
    parser.add_argument("--train_batch_size_per_gpu", default=64, type=int)
    parser.add_argument("--eval_batch_size_per_gpu", default=64, type=int)
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
    parser.add_argument("--eval_webnlg_dbpedia_file", default=None, type=str)
    parser.add_argument("--eval_gooaq_file", default=None, type=str)
    parser.add_argument("--eval_sq_file", default=None, type=str)
    parser.add_argument("--eval_mpww_file", default=None, type=str)
    parser.add_argument("--eval_mpww_passages_file", default=None, type=str)
    parser.add_argument("--eval_corpus_chunk_size", default=16384, type=int)
    # instrumentation
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--checkpoint_save_steps", default=1000, type=int)
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
    model = SentenceTransformer(args.model_name_or_path, add_pooling_layer=False)
    model.max_seq_length = args.max_seq_length
    word_embedding_model = model._first_module()
    word_embedding_model.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    train_datasets = [dataset_name for dataset_name in dataset_builders if
                      vars(args)[f"{dataset_name}_file"] is not None]
    name = args.run_name if args.run_name is not None else \
        f"{args.model_name_or_path.replace('/', '-')}_{'_'.join(train_datasets)}"

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
    sequential_evaluator, task_names = build_evaluators(args)

    if args.wandb and accelerator.is_main_process:

        import wandb

        wandb.init(project="rdf-embeddings", entity="flukeellington", name=name)
        wandb.config = {
            "model_name_or_path": args.model_name_or_path,
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
                if task_name == "MPWW":
                    wandb.log(
                        {f"{task_name}_accuracy@1": scores[i]["cos_sim"]["accuracy@k"][1], "training_steps": steps})
                    wandb.log({f"{task_name}_recall@10": scores[i]["cos_sim"]["recall@k"][10], "training_steps": steps})
                    wandb.log(
                        {f"{task_name}_precision@10": scores[i]["cos_sim"]["precision@k"][10], "training_steps": steps})
                    wandb.log({f"{task_name}_MRR@10": scores[i]["cos_sim"]["mrr@k"][10], "training_steps": steps})
                else:
                    wandb.log({f"{task_name}_recall@1": scores[i]["recall@1_src2tgt"], "training_steps": steps})
                    wandb.log({f"{task_name}_recall@10": scores[i]["recall@10_src2tgt"], "training_steps": steps})
                    wandb.log({f"{task_name}_MRR@10": scores[i]["mrr@10_src2tgt"], "training_steps": steps})

    else:
        train_callback = None
        eval_callback = None

    if args.num_epochs > 0:
        # Train the model
        model_save_path = os.path.join(args.output_dir, f'{name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        checkpoint_save_path = os.path.join(model_save_path, "checkpoints")
        os.makedirs(model_save_path, exist_ok=True)
        model.fit(train_objectives=[(dataloader, train_loss) for dataloader in dataloaders.values()],
                  evaluator=sequential_evaluator,
                  evaluation_steps=args.eval_steps,
                  epochs=args.num_epochs,
                  steps_per_epoch=args.steps_per_epoch,
                  warmup_steps=args.warmup_steps,
                  use_amp=False,
                  checkpoint_path=checkpoint_save_path,
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

    else:
        # trying to infer to which step went the model we're evaluating
        try:
            # reloading an existing model
            training_step = int(args.model_name_or_path.split("/")[-1])
            eval_path = os.path.join(args.model_name_or_path, "eval_out_of_training")
            os.makedirs(eval_path, exist_ok=True)
        except ValueError:
            # loading one from the hub
            training_step = -1
            eval_path = os.path.join(name, "eval_out_of_training")
            os.makedirs(eval_path, exist_ok=True)

        # evaluating
        for evaluator in sequential_evaluator.evaluators:
            evaluator.show_progress_bar = True
        main_score, scores = sequential_evaluator(model, output_path=eval_path, steps=training_step, epoch=1,
                                                  return_all_scores=True)

        # logging
        if args.wandb and accelerator.is_main_process:
            for i, task_name in enumerate(task_names):
                if task_name == "MPWW":
                    wandb.log({f"{task_name}_accuracy@1": scores[i]["cos_sim"]["accuracy@k"][1],
                               "training_steps": training_step})
                    wandb.log({f"{task_name}_recall@10": scores[i]["cos_sim"]["recall@k"][10],
                               "training_steps": training_step})
                    wandb.log(
                        {f"{task_name}_precision@10": scores[i]["cos_sim"]["precision@k"][10],
                         "training_steps": training_step})
                    wandb.log(
                        {f"{task_name}_MRR@10": scores[i]["cos_sim"]["mrr@k"][10], "training_steps": training_step})
                else:
                    wandb.log({f"{task_name}_recall@1": scores[i]["recall@1_src2tgt"], "training_steps": training_step})
                    wandb.log(
                        {f"{task_name}_recall@10": scores[i]["recall@10_src2tgt"], "training_steps": training_step})
                    wandb.log({f"{task_name}_MRR@10": scores[i]["mrr@10_src2tgt"], "training_steps": training_step})
