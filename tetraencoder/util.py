from contextlib import contextmanager
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim, mismatched_sizes_all_gather
from torch.nn import functional as F


def pair_sims_datasets_map(examples: dict, rank: int = 0, model: SentenceTransformer = None, text_key: str = "text",
                           rdf_key: str = "rdf_linearized", batch_size: int = 16, similarity_key: str = "similarity"):
    if rank is None:
        device = "cuda:0"
    else:
        device = f"cuda:{rank % torch.cuda.device_count()}"
    if isinstance(model, SentenceTransformer):
        model = model.to(device)
        embeddings1 = torch.stack(
            model.encode(examples[text_key], show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size,
                         device=device))
        embeddings2 = torch.stack(
            model.encode(examples[rdf_key], show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size,
                         device=device))

        cos_sims = F.cosine_similarity(embeddings1, embeddings2).detach().cpu().numpy()
        examples[similarity_key] = cos_sims

    elif isinstance(model, CrossEncoder):
        predictions = model.predict(list(zip(examples[text_key], examples[rdf_key])), batch_size=batch_size)
        examples[similarity_key] = predictions

    else:
        raise ValueError

    return examples


def all_embed_datasets_map(examples: dict, rank: int = 0, model: SentenceTransformer = None, text_key: str = "text",
                           rdf_key: str = "rdf_linearized", batch_size: int = 16):
    device = f"cuda:{rank % torch.cuda.device_count()}"
    model = model.to(device)
    embeddings1 = torch.stack(
        model.encode(examples[text_key], show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size,
                     device=device))
    embeddings2 = torch.stack(
        model.encode(examples[rdf_key], show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size,
                     device=device))

    examples["text_embedding"] = embeddings1.detach().cpu().numpy()
    examples["rdf_embedding"] = embeddings2.detach().cpu().numpy()

    return examples


def all_sims(dataset1: List[str], dataset2: List[str], model: SentenceTransformer, batch_size: int):
    if torch.distributed.is_initialized():
        # axis 0 (dataset1) of variable size, axis 1 (dataset2) of constant size
        # computing the second embeddings normally for everyone
        embeddings2 = torch.stack(
            model.encode(dataset2, show_progress_bar=False, convert_to_numpy=False, batch_size=batch_size))

        # using the first embeddings as they come to compute the similarity to avoid GPU errors
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sizes = [len(dataset1) // world_size + (1 if rank < len(dataset1) % world_size else 0)
                 for rank in range(world_size)]
        # dividing the list of sentences into batches
        limits = np.cumsum([0] + sizes)
        local_sentences = dataset1[limits[rank]:limits[rank + 1]]
        # embedding
        local_sims = []
        for start_index in range(0, len(local_sentences), batch_size):
            sentences_batch = local_sentences[start_index:start_index + batch_size]
            batch_embeddings = model._encode(sentences_batch, device=model._target_device)
            batch_sims = cos_sim(batch_embeddings, embeddings2)
            local_sims.append(batch_sims)
        local_sims = torch.cat(local_sims)
        cos_sims = torch.cat(mismatched_sizes_all_gather(local_sims))

    else:
        embeddings1 = model.encode(dataset1, show_progress_bar=False, convert_to_tensor=True, batch_size=batch_size,
                         num_proc=torch.cuda.device_count())
        embeddings2 = model.encode(dataset2, show_progress_bar=False, convert_to_tensor=True, batch_size=batch_size,
                         num_proc=torch.cuda.device_count())
        cos_sims = cos_sim(embeddings1, embeddings2)

    return cos_sims


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result
