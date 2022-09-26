import argparse
import os
from collections import OrderedDict
from functools import partial

import torch
from multiprocess import set_start_method

from sentence_transformers import SentenceTransformer, CrossEncoder

from dataset_wrappers import *
from util import pair_sims_datasets_map, all_embed_datasets_map


def autoload_model(model_name_or_path, **kwargs):
    if "cross" in model_name_or_path:
        return CrossEncoder(model_name_or_path)
    return SentenceTransformer(model_name_or_path, **kwargs)


def column_to_keep(column_name):
    return column_name == "text" or column_name == "rdf_linearized" or column_name.startswith("similarity")


if __name__ == "__main__":
    # CUDA multiprocessing
    set_start_method("spawn")

    # args
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--full_embeddings", action="store_true",
                        help="will return full embeddings rather than just similarity scores. Heavy disk use.")
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--model_names_or_paths", default=None, type=str, help="Pass a txt file with several models")
    parser.add_argument("--batch_size", default=128, type=int)
    # dataset args
    for dataset_name in dataset_builders:
        parser.add_argument(f"--{dataset_name}_file", default=None, type=str)
    parser.add_argument("--subset", default=None, type=int)
    parser.add_argument("--output_folder", default="outputs/dataset_embeddings", type=str)
    args = parser.parse_args()

    # setup
    os.makedirs(args.output_folder, exist_ok=True)
    if args.model_name_or_path is not None:
        assert args.model_names_or_paths is None, "can't pass both arguments"
        models = {args.model_name_or_path: autoload_model(args.model_name_or_path, add_pooling_layer=False)}
    else:
        assert args.model_names_or_paths is not None, "have to pass one argument"
        models = {name_or_path[:-1]: autoload_model(name_or_path[:-1]) for name_or_path in
                  open(args.model_names_or_paths).readlines()}
    datasets_to_embed = OrderedDict()
    for dataset_name in dataset_builders:
        dataset_file = vars(args)[f"{dataset_name}_file"]
        if dataset_file is not None:
            datasets_to_embed[dataset_name] = dataset_builders[dataset_name](dataset_file, setup_for_training=False)

    # main loop
    for dataset_name, dataset in datasets_to_embed.items():
        out_path = os.path.join(args.output_folder, dataset_name)
        if args.subset is not None:
            out_path = out_path + f"_subset{args.subset}"
            dataset.shuffle(seed=1066)
            dataset.select(range(args.subset))
        if args.full_embeddings:
            out_path = out_path + "_all_embed"
        out_path = out_path + ".jsonl"
        for name, model in models.items():
            if args.full_embeddings:
                assert not isinstance(model, CrossEncoder), "full embeddings don't work with CrossEncoders"
                dataset.map(
                    partial(all_embed_datasets_map, model=model, text_key="text", rdf_key="rdf_linearized",
                            batch_size=args.batch_size, similarity_key=f"similarity_{name}"),
                    batched=True, batch_size=args.batch_size, with_rank=True, num_proc=torch.cuda.device_count())
            else:
                dataset.map(
                    partial(pair_sims_datasets_map, model=model, text_key="text", rdf_key="rdf_linearized",
                            batch_size=args.batch_size, similarity_key=f"similarity_{name}"),
                    batched=True, batch_size=args.batch_size, with_rank=True, num_proc=torch.cuda.device_count())
        dataset.dataset = dataset.dataset.remove_columns(
            [column for column in dataset.dataset.column_names if not column_to_keep(column)])
        dataset.to_json(out_path)
