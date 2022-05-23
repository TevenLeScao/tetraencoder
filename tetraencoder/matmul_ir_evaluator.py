import logging
import os
import random
from functools import partial
from typing import List, Tuple, Dict, Set, Union

import numpy as np
import torch
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch import multiprocessing as mp
from torch.nn import functional as F

logger = logging.getLogger(__name__)
event = None


class MatmulIREvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 queries: Dict[str, str],  # qid => query
                 corpus: Dict[str, str],  # cid => doc
                 relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
                 index_training_samples: int = 4*4096,
                 pca_dims: int = 64,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [10],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 seed=1623,
                 name: str = '',
                 write_csv: bool = True,
                 score_function: str = 'cos_sim',  # Score function, higher=more similar
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        random.shuffle(self.corpus_ids)
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.index_training_samples = index_training_samples
        self.pca_dims = pca_dims
        # identity for now, will change if pca_dims != None

        self.relevant_docs = relevant_docs
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.seed = seed
        self.name = name
        self.write_csv = write_csv
        self.score_function = score_function

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for k in accuracy_at_k:
            self.csv_headers.append("MPWW_accuracy@{}".format(k))

        for k in precision_recall_at_k:
            self.csv_headers.append("MPWW_precision@{}".format(k))
            self.csv_headers.append("MPWW_recall@{}".format(k))

        for k in mrr_at_k:
            self.csv_headers.append("MPWW_MRR@{}".format(k))

        for k in ndcg_at_k:
            self.csv_headers.append("MPWW_NDCG@{}".format(k))

        for k in map_at_k:
            self.csv_headers.append("MPWW_MAP@{}".format(k))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, num_proc: int = None,
                 return_all_scores: bool = False, *args, **kwargs) -> Union[float, Tuple[float, dict]]:

        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch,
                                                                                                                 steps)
        else:
            out_txt = ":"

        if num_proc is None:
            num_proc = torch.cuda.device_count()

        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        if torch.distributed.is_initialized():
            raise NotImplementedError("this shouldn't be used in distributed mode")
        else:
            scores = self.compute_metrices(model, *args, num_proc=num_proc, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for k in self.accuracy_at_k:
                output_data.append(scores['accuracy@k'][k])

            for k in self.precision_recall_at_k:
                output_data.append(scores['precision@k'][k])
                output_data.append(scores['recall@k'][k])

            for k in self.mrr_at_k:
                output_data.append(scores['mrr@k'][k])

            for k in self.ndcg_at_k:
                output_data.append(scores['ndcg@k'][k])

            for k in self.map_at_k:
                output_data.append(scores['map@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        main_score = scores['map@k'][max(self.map_at_k)]
        if return_all_scores:
            return main_score, scores
        else:
            return main_score

    def compute_metrices(self, model: SentenceTransformer, corpus_model: SentenceTransformer = None, num_proc=None):

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k),
                    max(self.map_at_k))

        nc = len(self.corpus)  # corpus size
        nq = len(self.queries)  # nb of queries

        if corpus_model is None:
            corpus_model = model

        print("embedding the corpus")
        corpus_embeddings = MultiGPUEmbeddings(model, self.corpus, pca_dims=self.pca_dims, corpus_model=corpus_model,
                                               normalize=self.score_function == "cos_sim", pca_training_points=self.index_training_samples)

        # now let's search!
        print("searching for queries")
        queries_results_unnamed = corpus_embeddings.search(self.queries, top_k=max_k, batch_size=self.batch_size)
        queries_results_formatted = [
            [{"score": score, "corpus_id": corpus_id} for score, corpus_id in zip(query_scores, query_ids)] for
            query_scores, query_ids in zip(queries_results_unnamed)]

        logger.info("Queries: {}".format(nq))
        logger.info("Corpus: {}\n".format(nc))

        # Compute scores
        print("computing scores")
        scores = self.scores_from_results(queries_results_formatted)

        # Output
        logger.info("Score-Function: {}".format(self.score_function))
        self.output_scores(scores)

        return scores

    def scores_from_results(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in
                                       top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances,
                                                                                                       k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg,
                'mrr@k': MRR, 'map@k': AveP_at_k}

    def output_scores(self, scores):
        for k in scores['accuracy@k']:
            logger.info("accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k] * 100))

        for k in scores['precision@k']:
            logger.info("precision@{}: {:.2f}%".format(k, scores['precision@k'][k] * 100))

        for k in scores['recall@k']:
            logger.info("recall@{}: {:.2f}%".format(k, scores['recall@k'][k] * 100))

        for k in scores['mrr@k']:
            logger.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))

        for k in scores['ndcg@k']:
            logger.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))

        for k in scores['map@k']:
            logger.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg


class MultiGPUEmbeddings:

    def __init__(self, query_model, corpus, pca_dims=None, corpus_model=None, num_gpus=torch.cuda.device_count(),
                 batch_size=256, pca_training_points=16384, normalize=False):
        self.query_model = query_model
        self.corpus_model = query_model if corpus_model is None else corpus_model
        self.num_gpus = num_gpus
        self.normalize = normalize
        self.indexes = [None for _ in range(num_gpus)]
        self.corpuses = [[] for _ in range(num_gpus)]

        if pca_dims is None:
            self.d = self.corpus_model.encode(corpus[0]).shape[-1]  # dimension
            self.pca = None
        else:
            self.d = pca_dims
            random_sample = random.sample(corpus, pca_training_points)
            print("temporary embeddings as training data for the PCA")
            embeddings = self.corpus_model.encode(random_sample, convert_to_tensor=True, num_proc=num_gpus,
                                                        batch_size=batch_size)
            print("learning the PCA")
            self.pca = self.learn_pca(embeddings)

        self.create_index(corpus, batch_size=batch_size)

    def learn_pca(self, embeddings):

        avg = torch.mean(embeddings, dim=0, keepdim=True)
        embeddings = embeddings - avg
        _, _, V = torch.svd(embeddings)
        components = V[:, :self.d]
        return avg.cpu(), components.cpu()

    def create_index(self, corpus, batch_size=256):

        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        cuda_compatible_multiprocess = mp.get_context("spawn")
        with cuda_compatible_multiprocess.Pool(self.num_gpus) as p:
            for embeddings, rank, batch in tqdm(p.imap(partial(encode_and_pca,
                                                               model=self.corpus_model,
                                                               multiprocessing=True,
                                                               device=None,
                                                               pca=self.pca,
                                                               normalize=self.normalize),
                                                       batches),
                                                desc="embedding the corpus", total=len(batches)):
                if self.indexes[rank] is None:
                    self.indexes[rank] = embeddings
                else:
                    self.indexes[rank] = torch.cat((self.indexes[rank], embeddings))
                self.corpuses[rank].extend(batch)

    def compare_with_all_indexes(self, query_embeddings, top_k):
        cuda_compatible_multiprocess = mp.get_context("spawn")
        with cuda_compatible_multiprocess.Pool(self.num_gpus) as p:
            neighbours = p.map(partial(top_k_search, query_embeddings=query_embeddings, top_k=top_k), self.indexes)
        all_scores = torch.cat([neighbour[0] for neighbour in neighbours], dim=1)
        all_positions = torch.cat([neighbour[1] for neighbour in neighbours], dim=1)
        final_best_scores, all_meta_positions = torch.topk(all_scores, k=top_k, dim=1)

        def sentence_from_position(meta_position, row_positions):
            index_rank = meta_position // top_k
            within_index_position = row_positions[meta_position]
            return self.corpuses[index_rank][within_index_position]

        final_sentences = [[sentence_from_position(meta_position.item(), all_positions[row]) for meta_position in row_meta_positions] for row, row_meta_positions in
                           enumerate(all_meta_positions)]

        return final_best_scores.cpu().tolist(), final_sentences

    def search(self, queries, top_k, batch_size=256):

        scores, neighbour_sentences = [], []
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        cuda_compatible_multiprocess = mp.get_context("spawn")

        with cuda_compatible_multiprocess.Pool(self.num_gpus) as p:
            all_query_embeddings = tqdm(p.imap(partial(encode_and_pca,
                                                             model=self.corpus_model,
                                                             multiprocessing=True,
                                                             device=None,
                                                             pca=self.pca,
                                                             normalize=self.normalize), batches),
                                                   desc="searching for queries", total=len(batches))
            for query_embeddings, _, _ in all_query_embeddings:
                new_scores, new_sentences = self.compare_with_all_indexes(query_embeddings.cpu(), top_k=top_k)
                scores.extend(new_scores)
                neighbour_sentences.extend(new_sentences)

        return scores, neighbour_sentences


def encode_and_pca(batch, model, device=None, pca=None, multiprocessing=False, normalize=False):
    if multiprocessing:
        rank = mp.current_process()._identity[0]
        if device is None and torch.cuda.is_available():
            device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        rank = 0

    model.to(device)
    features = model.tokenize(batch)
    features = batch_to_device(features, device)

    with torch.no_grad():
        embeddings = model.forward(features)['sentence_embedding']
        if pca is not None:
            avg, components = pca
            embeddings = torch.matmul((embeddings - avg.to(embeddings.get_device())), components.to(embeddings.get_device()))
        if normalize:
            embeddings = F.normalize(embeddings)
        embeddings = embeddings.detach()

    return embeddings, rank % torch.cuda.device_count(), batch


def top_k_search(index, query_embeddings, top_k):
    with torch.no_grad():
        scores = torch.matmul(query_embeddings.to(index.get_device()), index.T)
        results = torch.topk(scores, k=top_k, dim=1)
    return results[0].cpu(), results[1].cpu()
