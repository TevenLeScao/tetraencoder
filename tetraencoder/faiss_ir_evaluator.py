import logging
import os
import random
from time import time, sleep
from typing import List, Tuple, Dict, Set, Union

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers.evaluation import SentenceEvaluator
from torch import multiprocessing as mp
from torch.nn import functional as F
from util import normalized

logger = logging.getLogger(__name__)
event = None


def embed_chunk(rank, model, corpus_chunk, batch_size, normalize, queue: mp.Queue, stop_signal: mp.Event):
    # The first (hence the +1 and -1) GPU is reserved for the index
    device_rank = rank % (torch.cuda.device_count() - 1) + 1
    device = f"cuda:{device_rank}"
    model = model.to(device)

    for batch in tqdm([corpus_chunk[start: start + batch_size] for start in range(0, len(corpus_chunk), batch_size)], desc=f"GPU {device_rank} chunks"):
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=False, convert_to_tensor=True, batch_size=batch_size, device=device)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        queue.put(embeddings)

    # Signal we are done
    queue.put(None)

    stop_signal.wait()


def add_chunk_to_index(index: faiss.Index, queue: mp.Queue, index_training_samples, num_workers, pca_dims=None, device=0, logging_interval=1000, corpus_size=-1):
    none_count = 0
    seen_examples = 0
    seen_chunks = 0
    initial_training_embeddings = []
    start_time = time()

    # identity for now, will change later if `pca_dims != None`
    pca = lambda embeddings: embeddings

    with tqdm(total=corpus_size, desc="Index adds") as pbar:
        while True:
            embeddings = queue.get()
            if embeddings is not None:
                seen_examples += embeddings.shape[0]
                seen_chunks += 1
                pbar.update(1)

                if logging_interval > 0 and seen_chunks % logging_interval == 0:
                    print(f"{seen_examples} examples out of {corpus_size} seen in {time() - start_time:.2f}s")

                embeddings = embeddings.to(f"cuda:{device}")
                if index.is_trained:
                    index.add(pca(embeddings))
                else:
                    initial_training_embeddings.append(embeddings)
                    if seen_examples > index_training_samples:
                        print("training index")
                        training_data = torch.cat(initial_training_embeddings)

                        if pca_dims is not None:
                            U, S, V = torch.pca_lowrank(training_data, q=pca_dims)
                            pca = lambda embeddings: torch.matmul(embeddings, V)

                        training_data = pca(training_data)
                        index.train(training_data)
                        print(f"index trained in {time() - start_time}s")
                    else:
                        continue
            else:
                none_count += 1
            if none_count == num_workers:
                print(f"Index built in {time() - start_time}")
                break


class FaissIREvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 index_training_samples: int = 40*4096,
                 index_nlist: int = 100,
                 index_nprobe: int = 5,
                 index_factory_string: str = "IVF4096,SQfp16", # IVF4096,Flat
                 pca_dims: int = 128,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [10],
                 show_progress_bar: bool = False,
                 corpus_chunk_size: int = 50000,
                 batch_size: int = 32,
                 seed=1623,
                 name: str = '',
                 write_csv: bool = True,
                 score_function: str = 'cos_sim',       #Score function, higher=more similar
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.index_training_samples = index_training_samples
        self.index_nlist = index_nlist
        self.index_nprobe = index_nprobe
        self.index_factory_string = index_factory_string
        self.pca_dims = pca_dims

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
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
            self.csv_headers.append("{}-Accuracy@{}".format(score_function, k))

        for k in precision_recall_at_k:
            self.csv_headers.append("{}-Precision@{}".format(score_function, k))
            self.csv_headers.append("{}-Recall@{}".format(score_function, k))

        for k in mrr_at_k:
            self.csv_headers.append("{}-MRR@{}".format(score_function, k))

        for k in ndcg_at_k:
            self.csv_headers.append("{}-NDCG@{}".format(score_function, k))

        for k in map_at_k:
            self.csv_headers.append("{}-MAP@{}".format(score_function, k))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, num_proc: int = None, return_all_scores: bool = False, *args, **kwargs) -> Union[float, Tuple[float, dict]]:

        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        if num_proc is None:
            num_proc = torch.cuda.device_count()

        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        if torch.distributed.is_initialized():
            # currenty this may hang because of index creation time
            scores = self.compute_metrices_distributed(model, *args, num_proc=num_proc, **kwargs)
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

    def compute_metrices(self, model, corpus_model=None, num_proc: int = None):

        gpu_mp = mp.get_context("spawn")
        queue = gpu_mp.Queue()
        stop_signal = gpu_mp.Event()
        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        if corpus_model is None:
            corpus_model = model

        if self.pca_dims is None:
            d = model.encode(self.queries[0]).shape[-1]  # dimension
        else:
            d = self.pca_dims
        nc = len(self.corpus)  # corpus size
        assert nc >= self.index_training_samples, f"corpus size {nc} is smaller than necessary training samples {self.index_training_samples}"
        nq = len(self.queries)  # nb of queries
        np.random.seed(1234)  # make reproducible
        # quantizer = faiss.IndexFlatL2(d)  # the other
        # metric = faiss.METRIC_L2
        # index = faiss.IndexIVFFlat(quantizer, d, self.index_nlist, metric)
        index_device_rank = 0
        index = faiss.index_factory(d, self.index_factory_string)
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, index_device_rank, index)
        index.nprobe = self.index_nprobe

        # consumer = gpu_mp.Process(target=add_chunk_to_index, args=(index, queue, self.index_training_samples, num_proc))
        # consumer.start()

        processes = []
        random_index = list(range(nc))
        random.shuffle(random_index)
        chunk_size = nc // (num_proc - 1) + 1
        chunk_idxes = [random_index[i * chunk_size:(i + 1) * chunk_size] for i in
                       range(num_proc - 1)]
        for rank in range(num_proc - 1):
            corpus_chunk = [self.corpus[idx] for idx in chunk_idxes[rank]]
            process = gpu_mp.Process(target=embed_chunk,
                              kwargs={"rank": rank, "model": corpus_model, "corpus_chunk": corpus_chunk,
                                      "batch_size": self.batch_size, "normalize": self.score_function == "cos_sim",
                                      "queue": queue, "stop_signal": stop_signal})

            processes.append(process)
            process.start()

        add_chunk_to_index(index, queue, self.index_training_samples, num_proc - 1, pca_dims=self.pca_dims, corpus_size=nc // self.batch_size, device=index_device_rank)

        stop_signal.set()
        for process in processes:
            process.join()

        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar,
                                        batch_size=self.batch_size, convert_to_numpy=True, num_proc=num_proc, multiprocessing_devices=list(range(1, torch.cuda.device_count())))
        # query_embeddings = query_embeddings.astype('float32')
        if self.score_function == "cos_sim":
            query_embeddings = normalized(query_embeddings, axis=1)

        # now let's search!
        queries_result_list = []
        for i in range(0, nq, self.batch_size):
            query_batch = query_embeddings[i:i+self.batch_size]
            scores, neighbours = index.search(query_batch, max_k)
            queries_result_list.extend([[{"corpus_id": corpus_id, "score": score} for corpus_id, score in zip(neighbours[j], scores[j])] for j in range(len(query_batch))])

        logger.info("Queries: {}".format(nq))
        logger.info("Corpus: {}\n".format(nc))

        #Compute scores
        scores = self.scores_from_results(queries_result_list)

        #Output
        logger.info("Score-Function: {}".format(self.score_function))
        self.output_scores(scores)

        return scores

    def compute_metrices_distributed(self, model, corpus_model=None, num_proc: int = None) -> Dict[str, float]:
        raise NotImplementedError

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
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
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


        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}


    def output_scores(self, scores):
        for k in scores['accuracy@k']:
            logger.info("accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))

        for k in scores['precision@k']:
            logger.info("precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))

        for k in scores['recall@k']:
            logger.info("recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))

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
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg
