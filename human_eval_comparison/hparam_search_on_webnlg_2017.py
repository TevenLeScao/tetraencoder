import argparse
import logging
import math
import os
import random

import datasets
import numpy as np
import torch
import wandb
from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses
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


def convert_2017_rdf(examples: dict, rdf_key: str = "mr", output_key="rdf"):
    examples[output_key] = [linearize_rdf([triple.split(" | ") for triple in example.split("<br>")]) for
                                        example in examples[rdf_key]]
    return examples


def train(config, data, metric_to_fit):

    train_samples = []
    dev_samples = []

    bounds = min(data[metric_to_fit]), max(data[metric_to_fit])

    def normalize_rating(score):
        return (score - bounds[0]) / (bounds[1] - bounds[0])

    data_2017 = data.map(
        lambda example: {k: normalize_rating(v) if k == metric_to_fit else v for k, v in example.items()})

    train_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9)))
    dev_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9), len(data_2017)))

    for i, row in enumerate(train_dataset):
        train_samples.append(InputExample(texts=[row['text'], row['rdf']],
                                          label=row[metric_to_fit]))

    for i, row in enumerate(dev_dataset):
        dev_samples.append(InputExample(texts=[row['text'], row['rdf']],
                                        label=row[metric_to_fit]))

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config["train_batch_size_per_gpu"])

    # We add an evaluator, which evaluates the performance during training
    evaluator = PearsonCorrelationEvaluator.from_input_examples(dev_samples, name=f"{metric_to_fit}_dev")

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)  # 10% of train data for warm-up
    # Train the model
    run = wandb.init(project="rdf-crossencoder", entity="flukeellington",
                     name=f"{'bi' if config['biencoder'] else 'cross'}_{config['model_name_or_path'].split('/')[-1]}_lr{config['lr']:.3}_bs{config['train_batch_size_per_gpu']}_{f'2020_{metric_to_fit}' if args.challenge_2020 else 2017}",
                     reinit=True,
                     config={
                         "learning_rate": config["lr"],
                         "epochs": NUM_EPOCHS,
                         "batch_size": config["train_batch_size_per_gpu"],
                         "biencoder": config["biencoder"],
                         "base_model": config["model_name_or_path"],
                         "year": 2020 if args.challenge_2020 else 2017,
                         "metric": metric_to_fit
                     })

    if config["biencoder"]:
        model = SentenceTransformer(config["model_name_or_path"], use_auth_token=True)
        train_loss = losses.CosineSimilarityLoss(model=model)

        def train_callback(score, epoch, steps):
            wandb.log(
                {"bi_encoder_loss": score.item(), "step": steps, "data_points": steps * train_dataloader.batch_size})

        def eval_callback(score, epoch, steps):
            wandb.log({"correlation": score, "epoch": epoch, "step": steps,
                       "data_points": steps * train_dataloader.batch_size})

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=NUM_EPOCHS,
                  warmup_steps=warmup_steps,
                  scheduler="warmupcosine",
                  output_path=config["output_path"],
                  optimizer_params={"lr": config["lr"]},
                  train_callback=train_callback,
                  eval_callback=eval_callback,
                  logging_steps=1,
                  full_scores_callbacks=False,
                  show_progress_bar=False,
                  early_stopping=EARLY_STOPPING)
        best_score = model.best_score

    else:
        model = BetterCrossEncoder(config["model_name_or_path"], num_labels=1)
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
                               output_path=config["output_path"],
                               early_stopping=EARLY_STOPPING)

    wandb.log({"best_correlation": best_score})
    run.finish()
    return best_score


if __name__ == "__main__":
    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="sentence-transformers/all-mpnet-base-v2",
                        type=str)  # "../outputs/small_bs_runs_allneg/all_bs160_allneg" "../outputs/small_bs_runs/all_datasets_bs320-2022-01-11_02-41-31"
    parser.add_argument("--challenge_2020", action="store_true")
    parser.add_argument("--metric", type=str, default="semantic_adequacy")
    parser.add_argument("--biencoder", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    search_space = {"lr": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
                    "train_batch_size_per_gpu": [8, 16, 32, 64, 96]}

    if args.challenge_2020:
        data = datasets.load_dataset("teven/webnlg_2020_human_eval")["train"].shuffle(seed=SEED)
        data = data.rename_column("data coverage", "data_coverage")
        data = data.add_column("metric_average", (np.array(data["correctness"]) + np.array(data["data_coverage"]) + np.array(data["relevance"])) / 3)

    else:
        data = datasets.load_dataset("teven/webnlg_2017_human_eval")["train"].filter(
            lambda example: example['text'] is not None and len(example['text']) > 0).shuffle(seed=SEED)
        data = data.map(convert_2017_rdf, batched=True)

    for lr in search_space["lr"]:
        for batch_size in search_space["train_batch_size_per_gpu"]:
            config = {"model_name_or_path": args.model_name_or_path, "lr": lr, "train_batch_size_per_gpu": batch_size,
                      "biencoder": args.biencoder, "metric": args.metric}
            config["output_path"] = f"hyperparam_search/{f'2020_{args.metric}' if args.challenge_2020 else 2017}/{'bi' if config['biencoder'] else 'cross'}_{config['model_name_or_path'].split('/')[-1]}/lr{config['lr']:.3}_bs{config['train_batch_size_per_gpu']}"
            if os.path.isdir(os.path.join(config["output_path"], "best_model")):
                print(f"-----------------\nSKIPPING run already found at {config['output_path']}\n-----------------")
                continue
            try:
                train(config, data=data, metric_to_fit=args.metric)
            except RuntimeError:
                pass

            if args.sanity:
                break
        if args.sanity:
            break
