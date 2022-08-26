import argparse
import logging
import math
import os
from datetime import datetime

import random

import numpy as np
import torch
import wandb

import datasets
from sentence_transformers import InputExample, LoggingHandler
from torch import nn
from torch.utils.data import DataLoader

from better_cross_encoder import BetterCrossEncoder, PearsonCorrelationEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

Q_TOKEN = "[Q]"
S_TOKEN = "[S]"
P_TOKEN = "[P]"
O_TOKEN = "[O]"
N_TOKEN = ""
SPECIAL_TOKENS = [Q_TOKEN, S_TOKEN, P_TOKEN, O_TOKEN, N_TOKEN]
CORRUPTION_BATCH_SIZE = 10000  # The higher the better
os.environ["WANDB_SILENT"] = "true"

NUM_EPOCHS = 50
EARLY_STOPPING = 5
SEED = 0


def wrap_triplets(examples, rdf_key):
    examples[rdf_key] = [[rdf] for rdf in examples[rdf_key]]
    return examples


def linearize_rdf(triples):
    encoded_rdf = ""
    for triple in triples:
        if len(triple) == 3:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {O_TOKEN} {triple[2]} "
        elif len(triple) == 4:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {triple[2]} {O_TOKEN} {triple[3]} "
        else:
            raise ValueError(f"Triple length was {len(triple)} instead of the expected 3 or 4")
    return encoded_rdf


def convert_2017_rdf(examples: dict, rdf_key: str = "mr"):
    examples[rdf_key + "_processed"] = [linearize_rdf([triple.split(" | ") for triple in example.split("<br>")]) for
                                        example in examples[rdf_key]]
    return examples


def train(config):
    assert torch.cuda.is_available()
    # Configure the training

    data_2017 = datasets.load_dataset("teven/webnlg_2017_human_eval")["train"].filter(
        lambda example: example['text'] is not None and len(example['text']) > 0).shuffle(seed=SEED)
    data_2017 = data_2017.map(convert_2017_rdf, batched=True)

    train_samples = []
    dev_samples = []

    bounds = min(data_2017["semantic_adequacy"]), max(data_2017["semantic_adequacy"])

    def normalize_rating(score):
        return (score - bounds[0]) / (bounds[1] - bounds[0])

    data_2017 = data_2017.map(
        lambda example: {k: normalize_rating(v) if k == "semantic_adequacy" else v for k, v in example.items()})

    train_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9)))
    dev_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9), len(data_2017)))

    for i, row in enumerate(train_dataset):
        train_samples.append(InputExample(texts=[row['text'], row['mr_processed']],
                                          label=row['semantic_adequacy']))

    for i, row in enumerate(dev_dataset):
        dev_samples.append(InputExample(texts=[row['text'], row['mr_processed']],
                                        label=row['semantic_adequacy']))

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config["train_batch_size_per_gpu"])

    # We add an evaluator, which evaluates the performance during training
    evaluator = PearsonCorrelationEvaluator.from_input_examples(dev_samples, name='sem_adequacy_dev')

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)  # 10% of train data for warm-up

    model = BetterCrossEncoder(config["model_name_or_path"], num_labels=1, device="cuda")
    # Train the model
    run = wandb.init(project="rdf-crossencoder", entity="flukeellington",
                     name=f"{config['model_name_or_path'].split('/')[-1]}_lr{config['lr']:.3}_bs{config['train_batch_size_per_gpu']}",
                     reinit=True)
    best_score = model.fit(train_dataloader=train_dataloader,
                           evaluator=evaluator,
                           epochs=NUM_EPOCHS,
                           warmup_steps=warmup_steps,
                           scheduler="warmupcosine",
                           optimizer_params={"lr": config["lr"]},
                           evaluation_steps=len(train_dataloader),
                           activation_fct=nn.Sigmoid(),
                           loss_fct=nn.SmoothL1Loss(),
                           show_progress_bar=False,
                           log_wandb=True,
                           output_path=f"{config['model_name_or_path'].split('/')[-1]}/lr{config['lr']:.3}_bs{config['train_batch_size_per_gpu']}",
                           early_stopping=EARLY_STOPPING)
    wandb.log({"best_correlation": best_score})
    run.finish()
    return best_score


if __name__ == "__main__":
    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="sentence-transformers/all-mpnet-base-v2",
                        type=str)  # "../outputs/small_bs_runs_allneg/all_bs160_allneg" "../outputs/small_bs_runs/all_datasets_bs320-2022-01-11_02-41-31"
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    search_space = {"lr": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
                    "train_batch_size_per_gpu": [8, 16, 32, 64, 128]}

    for lr in search_space["lr"]:
        for batch_size in search_space["train_batch_size_per_gpu"]:
            config = {"model_name_or_path": args.model_name_or_path, "lr": lr, "train_batch_size_per_gpu": batch_size}
            train(config)
