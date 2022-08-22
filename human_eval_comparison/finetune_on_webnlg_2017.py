import argparse
import logging
import math
from datetime import datetime

import datasets
import wandb
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name_or_path", default="sentence-transformers/all-mpnet-base-v2",
                        type=str)  # "../outputs/small_bs_runs_allneg/all_bs160_allneg" "../outputs/small_bs_runs/all_datasets_bs320-2022-01-11_02-41-31"
    # training args
    parser.add_argument("--train_batch_size_per_gpu", default=64, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    # instrumentation
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--scheduler", default="warmupcosine",
                        choices=["warmupcosine", "warmuplinear", "warmupcosinewithhardrestarts"])
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    print(args)

    model_save_path = f'webnlg_cross_encoders/{args.run_name}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    model = BetterCrossEncoder(args.model_name_or_path, num_labels=1)

    data_2017 = datasets.load_dataset("csv", data_files="raw_data/2017_scores.csv")["train"].filter(
        lambda example: example['text'] is not None).shuffle()
    data_2017 = data_2017.map(convert_2017_rdf, batched=True)
    if args.sanity:
        data_2017 = data_2017.select(range(40))

    train_samples = []
    dev_samples = []

    bounds = min(data_2017["semantic_adequacy"]), max(data_2017["semantic_adequacy"])
    def normalize_rating(score):
        return (score - bounds[0]) / (bounds[1] - bounds[0])
    data_2017 = data_2017.map(lambda example: {k: normalize_rating(v) if k == "semantic_adequacy" else v for k, v in example.items()})

    train_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9)))
    dev_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9), len(data_2017)))

    for i, row in enumerate(train_dataset):
        train_samples.append(InputExample(texts=[row['text'], row['mr_processed']],
                                          label=row['semantic_adequacy']))

    for i, row in enumerate(dev_dataset):
        dev_samples.append(InputExample(texts=[row['text'], row['mr_processed']],
                                        label=row['semantic_adequacy']))

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size_per_gpu)

    # We add an evaluator, which evaluates the performance during training
    evaluator = PearsonCorrelationEvaluator.from_input_examples(dev_samples, name='sem_adequacy_dev')

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up

    if args.wandb:
        wandb.init(project="rdf-crossencoder", entity="flukeellington", name=args.run_name)
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.train_batch_size_per_gpu,
        }

    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=args.num_epochs,
              warmup_steps=warmup_steps,
              scheduler=args.scheduler,
              output_path=model_save_path,
              optimizer_params={"lr": args.lr},
              evaluation_steps=len(train_dataloader),
              activation_fct=nn.Sigmoid(),
              loss_fct=nn.SmoothL1Loss(),
              log_wandb=args.wandb)

    ##### Load model and eval on test set
    model = BetterCrossEncoder(model_save_path)
    evaluator(model)
