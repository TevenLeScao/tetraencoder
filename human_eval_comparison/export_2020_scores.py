import json
import os
from collections import OrderedDict

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from scipy.stats import spearmanr

SCORE_FILE = "2020_scores.json"


def pair_sims(model: SentenceTransformer, texts, rdfs, batch_size: int = 32):
    device = f"cuda"
    model = model.to(device)
    embeddings1 = torch.stack(
        model.encode(texts, show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size, device=device))
    embeddings2 = torch.stack(
        model.encode(rdfs, show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size, device=device))

    cos_sims = F.cosine_similarity(embeddings1, embeddings2).detach().cpu().numpy()

    return cos_sims


def cross_sims(model: CrossEncoder, texts, rdfs, batch_size: int = 16):
    predictions = model.predict(list(zip(texts, rdfs)), batch_size=batch_size)
    return predictions


if __name__ == "__main__":
    candidate_folders = list_subfolders_with_paths = [f.path for f in os.scandir("raw_data/en_submissions_rdf2text") if
                                                      f.is_dir()]

    human_scores = json.load(open("raw_data/english_humeval_data_all_teams.json"))

    teams = set([item["submission_id"] for item in human_scores])
    human_scores_per_team = {team: [] for team in teams}
    for item in human_scores:
        human_scores_per_team[item["submission_id"]].append(item)

    # the last line is inconsistent, let's just fix this
    rdfs = open(os.path.join("raw_data", "unwrapped_rdfs.txt")).readlines()[:1779]
    ticks = ["correctness", "data coverage", "relevance", "fluency", "text structure", "bert precision",
             "bert recall", "bert F1", "bleurt", "bleu"]

    if os.path.exists(SCORE_FILE):
        data_2020 = json.load(open(SCORE_FILE))
    else:
        data_2020 = {tick: [] for tick in ticks}
        for candidate_folder in tqdm(candidate_folders):
            team = candidate_folder.split("/")[-1]
            print(team)
            all_auto_scores = json.load(open(os.path.join(candidate_folder, "primary.en_results")))
            all_bleurt_scores = json.load(open(os.path.join(candidate_folder, "primary.en_results_bleurt")))
            all_hypotheses = open(os.path.join(candidate_folder, "primary.en")).readlines()[:1779]

            for item in human_scores_per_team[team]:
                sample_id = int(item["sample_id"]) - 1
                # human judgment
                data_2020["correctness"].append(item["Correctness"])
                data_2020["data coverage"].append(item["DataCoverage"])
                data_2020["fluency"].append(item["Fluency"])
                data_2020["relevance"].append(item["Relevance"])
                data_2020["text structure"].append(item["TextStructure"])
                # automatic scores
                data_2020["bert precision"].append(all_auto_scores["bert_precision"][sample_id])
                data_2020["bert recall"].append(all_auto_scores["bert_recall"][sample_id])
                data_2020["bert F1"].append(all_auto_scores["bert_f1"][sample_id])
                data_2020["bleurt"].append(all_bleurt_scores["bleurt"][sample_id])
                data_2020["bleu"].append(all_auto_scores["bleu_nltk"][sample_id])
            # data_2020["ter"].append(all_auto_scores["ter"][sample_id])

    rdfs = open(os.path.join("raw_data", "unwrapped_rdfs.txt")).readlines()[:1779]

    hparam_search_results = open("best_models_hparam_search.txt").readlines()
    models = OrderedDict([
                             ("all_bs160_allneg", SentenceTransformer(
                                 "teven/all_bs160_allneg")),
                             ("all_bs192_hardneg", SentenceTransformer(
                                 "teven/all_bs192_hardneg")),
                             ("cross_all_bs160_allneg", CrossEncoder(
                                 "output/allneg_good_outlier"))
                         ] + [
                             (f"finetuned_{path.split('/')[0]}", CrossEncoder(
                                 f"search_results/{path[:-1]}/best_model")) for path in hparam_search_results if
                             "cross_" in path
                         ] + [
                             (f"finetuned_{path.split('/')[0]}", SentenceTransformer(
                                 f"search_results/{path[:-1]}/best_model")) for path in hparam_search_results if
                             "bi_" in path
                         ])

    for model_name, model in models.items():
        if model_name not in data_2020.keys():
            data_2020[model_name] = []
            ticks.append(model_name)

            for candidate_folder in tqdm(candidate_folders):
                team = candidate_folder.split("/")[-1]
                all_hypotheses = open(os.path.join(candidate_folder, "primary.en")).readlines()[:1779]

                if isinstance(model, CrossEncoder):
                    all_sim_scores = cross_sims(model, all_hypotheses, rdfs)
                else:
                    all_sim_scores = pair_sims(model, all_hypotheses, rdfs)

                for item in human_scores_per_team[team]:
                    sample_id = int(item["sample_id"]) - 1
                    # our score
                    data_2020[model_name].append(float(all_sim_scores[sample_id]))
                # automatic scores

    json.dump(data_2020, open(SCORE_FILE, "w"), indent=2, ensure_ascii=False)
