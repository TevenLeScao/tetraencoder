import argparse

import math
from scipy.stats import pearsonr
import datasets
import wandb

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
    # instrumentation
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    print(args)

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
    data_2017.map(lambda example: {k: normalize_rating(v) if k == "semantic_adequacy" else v for k, v in example.items()})

    train_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9)))
    dev_dataset = data_2017.select(range(math.floor(len(data_2017) * 0.9), len(data_2017)))

    if args.wandb:
        wandb.init(project="rdf-crossencoder", entity="flukeellington", name=args.run_name)
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.train_batch_size_per_gpu,
        }

    bleu_pearson = pearsonr(dev_dataset["bleu"], dev_dataset["semantic_adequacy"])
    meteor_pearson = pearsonr(dev_dataset["meteor"], dev_dataset["semantic_adequacy"])
    ter_pearson = pearsonr(dev_dataset["ter"], dev_dataset["semantic_adequacy"])
