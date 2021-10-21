import argparse
import os.path
from datetime import datetime
from time import time
import random
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import TranslationEvaluator
from torch.utils.data import DataLoader

from tetraencoder.dataset_builders import *

dataset_builders = {"msmarco": MsMarcoDataset, "kelm": KelmDataset, "gooaq": GooAqDataset, "tekgen": TekgenDataset}

if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--max_seq_length", default=300, type=int)
    # training args
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--steps_per_epoch", default=None, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--gradient_accumulation", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--checkpoint_save_steps", default=1000, type=int)
    # i/o args
    parser.add_argument("--output_dir", default=".", type=str)
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    # evaluation dataset args
    parser.add_argument(f"--eval_webnlg_wikidata_file", default=None, type=str)
    # wrap-up
    args = parser.parse_args()
    print(args)

    # Build model
    model = SentenceTransformer(args.model_name, add_pooling_layer=False)
    model.max_seq_length = args.max_seq_length
    word_embedding_model = model._first_module()
    word_embedding_model.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model_save_path = os.path.join(args.output_dir, f'train_bi-encoder-mnrl-{args.model_name.replace("/", "-")}-'
                                                    f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    # Build datasets
    source_data = {}
    dataloaders = {}
    eval_dataloaders = {}
    for dataset_name in dataset_builders:
        start_time = time()
        # for training
        input_filepath = vars(args)[f"{dataset_name}_file"]
        if input_filepath is not None:
            print(f"adding {dataset_name} to the corpus")
            dataloaders[dataset_name] = dataset_builders[dataset_name](input_filepath)
            dataloaders[dataset_name] = DataLoader(dataloaders[dataset_name], shuffle=False,
                                                   batch_size=args.train_batch_size)
            print(f"added {dataset_name} to the corpus in {time() - start_time:.3f}s")

    # Create evaluators
    if args.eval_webnlg_wikidata_file is not None:
        webnlg_dataset = WebNlgWikidataDataset(args.eval_webnlg_wikidata_file)
        evaluator = TranslationEvaluator(webnlg_dataset.rdfs(), webnlg_dataset.sentences(), show_progress_bar=True,
                                         batch_size=32)
    else:
        evaluator = None

    # Train the model
    model.fit(train_objectives=[(dataloader, train_loss) for dataloader in dataloaders.values()],
              evaluator=evaluator,
              evaluation_steps=10,
              epochs=args.num_epochs,
              steps_per_epoch=args.steps_per_epoch,
              warmup_steps=args.warmup_steps,
              use_amp=False,
              checkpoint_path=model_save_path,
              checkpoint_save_steps=args.checkpoint_save_steps,
              optimizer_params={'lr': args.lr},
              gradient_accumulation=args.gradient_accumulation
              )

    # Save the model
    model.save(model_save_path)
