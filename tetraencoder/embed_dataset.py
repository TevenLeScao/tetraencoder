import argparse
import os
from collections import OrderedDict
from functools import partial

import torch
from multiprocess import set_start_method

from sentence_transformers import SentenceTransformer

from dataset_wrappers import *
from util import pair_sims_datasets_map


dataset_builders = OrderedDict([("genwiki", GenWikiDataset), ("tekgen", TekgenDataset), ("trex", TRexDataset), ("kelm", KelmDataset)])
text_keys = {"genwiki": GenWikiDataset, "kelm": KelmDataset, "tekgen": TekgenDataset, "trex": TRexDataset}

if __name__ == "__main__":
    # CUDA multiprocessing
    set_start_method("spawn")

    # args
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    parser.add_argument("--subset", default=None, type=int)
    parser.add_argument("--output_folder", default="outputs/dataset_embeddings", type=str)
    args = parser.parse_args()

    # setup
    os.makedirs(args.output_folder, exist_ok=True)
    model = SentenceTransformer(args.model_name_or_path, add_pooling_layer=False)
    datasets_to_embed = OrderedDict()
    for dataset_name in dataset_builders:
        dataset_file = vars(args)[f"{dataset_name}_file"]
        if dataset_file is not None:
            datasets_to_embed[dataset_name] = dataset_builders[dataset_name](dataset_file)

    # main loop
    for dataset_name, dataset in datasets_to_embed.items():
        out_path = os.path.join(args.output_folder, dataset_name)
        if args.subset is not None:
            out_path = out_path + f"_subset{args.subset}"
            dataset.shuffle(seed=1066)
            dataset.select(range(args.subset))
        out_path = out_path + ".jsonl"
        dataset.map(
            partial(pair_sims_datasets_map, model=model, text_key="text", rdf_key="rdf_linearized", batch_size=args.batch_size),
            batched=True, batch_size=args.batch_size, with_rank=True, num_proc=torch.cuda.device_count())
        dataset.to_json(out_path)
