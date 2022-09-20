import os

from sentence_transformers import SentenceTransformer

for year in [2017, 2020]:
    for name in open(f"best_models_hparam_search_{year}.txt").readlines():
        path = os.path.join("hyperparam_search", str(year), name[:-1], "best_model")
        name = name.split("/")[0]
        print(name)
        model = SentenceTransformer(path)
        model.save_to_hub(f"{name}_finetuned_WebNLG{year}", private=True)
