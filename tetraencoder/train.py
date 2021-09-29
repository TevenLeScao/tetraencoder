import argparse
from datetime import datetime
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from time import time


def ms_marco_dataset(input_filepath):
    with open(input_filepath, 'r') as f:
        data = [InputExample(texts=[json.loads(line)["texts"][0], json.loads(line)["texts"][1]]) for line in f]
    return data


dataset_builders = {"msmarco": ms_marco_dataset}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--max_seq_length", default=300, type=int)
    # training args
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--checkpoint_save_steps", default=1000, type=int)
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    # wrap-up
    args = parser.parse_args()
    print(args)

    # build model
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    # build datasets
    datasets = {}
    dataloaders = {}
    for dataset_name in dataset_builders:
        start_time = time()
        input_filepath = vars(args)[f"{dataset_name}_file"]
        if input_filepath is not None:
            print(f"adding {dataset_name} to the corpus")
            datasets[dataset_name] = dataset_builders[dataset_name](input_filepath)
            dataloaders[dataset_name] = DataLoader(datasets[dataset_name], shuffle=True,
                                                   batch_size=args.train_batch_size)
        print(f"added {dataset_name} to the corpus in {time() - start_time:.3f}s")

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model_save_path = f'output/train_bi-encoder-mnrl-{args.model_name.replace("/", "-")}-' \
                      f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    # Train the model
    model.fit(train_objectives=[(dataloader, train_loss) for dataloader in dataloaders.values()],
              epochs=args.num_epochs,
              warmup_steps=args.warmup_steps,
              use_amp=True,
              checkpoint_path=model_save_path,
              checkpoint_save_steps=args.checkpoint_save_steps,
              optimizer_params={'lr': args.lr},
              )

    # Save the model
    model.save(model_save_path)
