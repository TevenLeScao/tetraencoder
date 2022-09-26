import os.path
from functools import partial
from typing import Callable
from collections import OrderedDict

import datasets
from sentence_transformers import SentenceTransformer, CrossEncoder
from export_2020_scores import cross_sims, pair_sims, quest_sims

SCORE_FILE = "2017_scores.jsonl"
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


def sim_map(examples: dict, sim_func: Callable, text_key: str, rdf_key: str,
            batch_size: int = 32, output_key="similarity", default_value=0):
    good_indices = [i for i, text in enumerate(examples[text_key]) if text is not None]

    scores = sim_func(texts=[examples[text_key][i] for i in good_indices],
                      rdfs=[examples[rdf_key][i] for i in good_indices],
                      batch_size=batch_size)

    examples[output_key] = [default_value] * len(examples[text_key])

    for idx, score in zip(good_indices, scores):
        examples[output_key][idx] = score

    return examples


# def pair_sims_map(examples: dict, model: SentenceTransformer, text_key: str, rdf_key: str,
#                   batch_size: int = 32, output_key="similarity"):
#     device = f"cuda"
#     model = model.to(device)
#     embeddings1 = torch.stack(
#         model.encode(examples[text_key], show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size,
#                      device=device))
#     embeddings2 = torch.stack(
#         model.encode(examples[rdf_key], show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size,
#                      device=device))
#
#     cos_sims = F.cosine_similarity(embeddings1, embeddings2).detach().cpu().numpy()
#     examples[output_key] = cos_sims
#
#     return examples
#
#
# def questeval_map(examples: dict, linearizer, questeval: QuestEval, text_key: str, rdf_key: str,
#                   output_key="questeval"):
#     rdfs = [linearizer(rdf.split("<br>")) for rdf in examples[rdf_key]]
#     examples[output_key] = questeval.corpus_questeval(
#         hypothesis=examples[text_key],
#         sources=rdfs,
#     )["ex_level_scores"]
#
#     return examples


if __name__ == "__main__":
    ticks = ["semantics", "grammar", "fluency", "bleu", "meteor", "ter", "questeval"]

    if os.path.exists(SCORE_FILE):
        data_2017 = datasets.load_dataset("json", data_files=SCORE_FILE)["train"]
    else:
        data_2017 = datasets.load_dataset("csv", data_files="raw_data/2017_averaged_scores.csv")["train"]
    data_2017 = data_2017.map(convert_2017_rdf, batched=True)
    data_2017 = data_2017.map(lambda example: {k: -v if k == "ter" and v > 0 else v for k, v in example.items()})

    try:
        import spacy
        from questeval.questeval_metric import QuestEval
        from questeval.utils import LinearizeWebnlgInput

        try:
            spacy_pipeline = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download

            download('en_core_web_sm')
            spacy_pipeline = spacy.load('en_core_web_sm')

        linearizer = LinearizeWebnlgInput(spacy_pipeline=spacy_pipeline)
        questeval = QuestEval()

        data_2017 = data_2017.map(
            partial(sim_map,
                    sim_func=partial(quest_sims, questeval=questeval),
                    text_key="text", rdf_key="mr", output_key="questeval_vanilla"),
            batched=True, batch_size=32)

        questeval.task = "data2text"
        data_2017 = data_2017.map(
            partial(sim_map,
                    sim_func=partial(quest_sims, questeval=questeval),
                    text_key="text", rdf_key="mr", output_key="questeval_data2text"),
            batched=True, batch_size=32)

    except (ImportError, ModuleNotFoundError):
        pass

    base_models = [
        "all-mpnet-base-v2",
        "all_bs160_allneg",
        "all_bs320_vanilla",
        "all_bs192_hardneg",
    ]
    metrics =[
        "correctness",
        "data_coverage",
        "relevance",
        "metric_average",
    ]
    models = OrderedDict(
        [
            ("all_bs160_allneg", SentenceTransformer(
                "teven/all_bs160_allneg")),
            ("all_bs192_hardneg", SentenceTransformer(
                "teven/all_bs192_hardneg"))
        ] + [
            (f"bi_{name}_{metric}", SentenceTransformer(f"teven/bi_{name}_finetuned_WebNLG2020_{metric}")) for name in base_models for metric in metrics
        ] + [
            (f"cross_{name}_{metric}", CrossEncoder(f"teven/cross_{name}_finetuned_WebNLG2020_{metric}")) for name in base_models for metric in metrics
        ]
    )

    for model_name, model in models.items():
        if model_name not in data_2017.column_names:
            if "cross_" in model_name:
                data_2017 = data_2017.map(
                    partial(sim_map,
                            sim_func=partial(cross_sims, model=model),
                            text_key="text", rdf_key="mr_processed", output_key=model_name),
                    batched=True, batch_size=32)
            else:
                data_2017 = data_2017.map(
                    partial(sim_map,
                            sim_func=partial(pair_sims, model=model),
                            text_key="text", rdf_key="mr_processed", output_key=model_name),
                    batched=True, batch_size=32)

    data_2017.to_json(SCORE_FILE)
