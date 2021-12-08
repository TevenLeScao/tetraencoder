import csv
import logging
import os
from typing import List, Union, Tuple

import numpy as np
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import pytorch_cos_sim

logger = logging.getLogger(__name__)


class TranslationEvaluatorWithRecall(SentenceEvaluator):
    """
    Given two sets of sentences in different languages, e.g. (en_1, en_2, en_3...) and (fr_1, fr_2, fr_3, ...),
    and assuming that fr_i is the translation of en_i.
    Checks if vec(en_i) has the highest similarity to vec(fr_i). Computes the accurarcy in both directions
    """

    def __init__(self,
                 source_sentences: List[str],
                 target_sentences: List[str],
                 show_progress_bar: bool = False,
                 batch_size: int = 16,
                 name: str = '',
                 print_wrong_matches: bool = False,
                 write_csv: bool = True,
                 mrr_ks: List[int] = None,
                 recall_ks: List[int] = None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param source_sentences:
            List of sentences in source language
        :param target_sentences:
            List of sentences in target language
        :param print_wrong_matches:
            Prints incorrect matches
        :param write_csv:
            Write results to CSV file
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.print_wrong_matches = print_wrong_matches

        self.mrr_ks = [10] if mrr_ks is None else mrr_ks
        self.recall_ks = [1, 3, 5, 10] if recall_ks is None else recall_ks

        assert len(self.source_sentences) == len(self.target_sentences)

        if name:
            name = "_" + name

        self.csv_file = "translation_evaluation" + name + "_results.csv"
        self.output_names = [f"recall@{k}_src2tgt" for k in self.recall_ks] + \
                            [f"recall@{k}_tgt2src" for k in self.recall_ks] + \
                            [f"mrr@{k}_src2tgt" for k in self.mrr_ks] + \
                            [f"mrr@{k}_tgt2src" for k in self.mrr_ks]
        self.csv_headers = ["epoch", "steps"] + self.output_names
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1,
                 return_all_scores: bool = False) -> Union[Tuple[float, dict], float]:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluating translation matching Accuracy on " + self.name + " dataset" + out_txt)

        src_embeddings = torch.stack(
            model.encode(self.source_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size,
                         convert_to_numpy=False))
        tgt_embeddings = torch.stack(
            model.encode(self.target_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size,
                         convert_to_numpy=False))

        cos_sims = pytorch_cos_sim(src_embeddings, tgt_embeddings).detach().cpu().numpy()

        src2tgt_recall_scores = []
        tgt2src_recall_scores = []
        src2tgt_mrr_scores = []
        tgt2src_mrr_scores = []

        for k in self.recall_ks:
            src2tgt_recall_scores.append(self.recall_at_k(k, cos_sims, verbose=self.print_wrong_matches and k == 1))
        for k in self.mrr_ks:
            src2tgt_mrr_scores.append(self.mrr_at_k(k, cos_sims))

        cos_sims = cos_sims.T

        for k in self.recall_ks:
            tgt2src_recall_scores.append(self.recall_at_k(k, cos_sims))
        for k in self.mrr_ks:
            tgt2src_mrr_scores.append(self.mrr_at_k(k, cos_sims))

        acc_src2trg = src2tgt_recall_scores[0]
        acc_trg2src = tgt2src_recall_scores[0]

        logger.info("Accuracy src2trg: {:.2f}".format(acc_src2trg * 100))
        logger.info("Accuracy trg2src: {:.2f}".format(acc_trg2src * 100))

        outputs = src2tgt_recall_scores + tgt2src_recall_scores + src2tgt_mrr_scores + tgt2src_mrr_scores
        assert len(outputs) == len(self.output_names), \
            f"Mismatched output length {len(outputs)} and expected output length {len(self.output_names)}"

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps] + outputs)

        if return_all_scores:
            return (acc_src2trg + acc_trg2src) / 2, {self.output_names[i]: outputs[i] for i in range(len(outputs))}
        else:
            return (acc_src2trg + acc_trg2src) / 2

    def recall_at_k(self, k, cos_sims, verbose=False):

        correct_src2trg = 0
        max_idxes = np.argpartition(cos_sims, -k, axis=1)[:, -k:]
        for i in range(len(cos_sims)):
            max_idx = max_idxes[i]
            if i in max_idx:
                correct_src2trg += 1
            elif verbose:
                print("i:", i, "j:", max_idx, "INCORRECT" if i in max_idx else "CORRECT")
                print("Src:", self.source_sentences[i])
                print("Trg:", self.target_sentences[max_idx[0]])
                print("Argmax score:", cos_sims[i][max_idx[0]], "vs. correct score:", cos_sims[i][i])

                results = zip(range(len(cos_sims[i])), cos_sims[i])
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for idx, score in results[0:5]:
                    print("\t", idx, "(Score: %.4f)" % (score), self.target_sentences[idx])

        return correct_src2trg / len(cos_sims)

    def mrr_at_k(self, k, cos_sims, verbose=False):

        max_idxes = np.argpartition(cos_sims, -k, axis=1)[:, -k:]
        sorted_max_idxes = np.take_along_axis(max_idxes, np.argsort(np.take_along_axis(cos_sims, max_idxes, axis=1)),
                                              axis=1)
        total_reciprocal_rank = 0
        for i in range(len(cos_sims)):
            sorted_max_idx = sorted_max_idxes[i]
            if i in sorted_max_idx:
                total_reciprocal_rank += 1 / (k - sorted_max_idx.tolist().index(i))
            elif verbose:
                print("i:", i, "j:", sorted_max_idx, "INCORRECT" if i in sorted_max_idx else "CORRECT")
                print("Src:", self.source_sentences[i])
                print("Trg:", self.target_sentences[sorted_max_idx[-1]])
                print("Argmax score:", cos_sims[i][sorted_max_idx[-1]], "vs. correct score:", cos_sims[i][i])

                results = zip(range(len(cos_sims[i])), cos_sims[i])
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for idx, score in results[0:5]:
                    print("\t", idx, "(Score: %.4f)" % (score), self.target_sentences[idx])

        return total_reciprocal_rank / len(cos_sims)
