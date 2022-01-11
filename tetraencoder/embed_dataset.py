import argparse
from functools import partial
from multiprocess import set_start_method

from sentence_transformers import SentenceTransformer

from dataset_builders import GenWikiDataset, TRexDataset
from util import pair_sims_datasets_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument(f"--batch_size", default=16, type=int)
    # dataset args
    parser.add_argument("--gen_wiki_dataset", default=None, type=str)
    parser.add_argument("--trex_dataset", default=None, type=str)
    parser.add_argument("--subset", default=None, type=int)
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)
    if args.trex_dataset:
        dataset = TRexDataset(args.trex_dataset)
        text_key = "text"
        rdf_key = "rdf_linearized"
        out_path = args.trex_dataset + ".embed"
    elif args.gen_wiki_dataset:
        dataset = GenWikiDataset(args.gen_wiki_dataset)
        text_key = "text"
        rdf_key = "rdf_linearized"
        out_path = args.gen_wiki_dataset + ".embed"
    else:
        raise NotImplementedError("you must pass a dataset!")
    if args.subset:
        out_path = out_path + f".subset{args.subset}"
        dataset = dataset.shuffle(seed=1066)
        dataset = dataset.select(range(args.subset))
    set_start_method("spawn")
    dataset = dataset.map(
        partial(pair_sims_datasets_map, model=model, text_key=text_key, rdf_key=rdf_key, batch_size=args.batch_size),
        batched=True, batch_size=args.batch_size, with_rank=True, num_proc=2)
    dataset.to_json(out_path)
