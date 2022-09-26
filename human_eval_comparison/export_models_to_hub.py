import os

import requests
from sentence_transformers import SentenceTransformer

for year in ["2017", "2020"]:
    for name in open(f"best_models_hparam_search_{year}.txt").readlines():
        path = os.path.join("hyperparam_search", name[:-1], "best_model")
        name = path.split("/")[2]
        year_and_metric = path.split("/")[1]
        model = SentenceTransformer(path)
        try:
            model.save_to_hub(f"{name}_finetuned_WebNLG{year_and_metric}")
            print(f"exporting model {name}_finetuned_WebNLG{year_and_metric}")
        except requests.exceptions.HTTPError:
            print(f"model {name}_finetuned_WebNLG{year_and_metric} already exists")
