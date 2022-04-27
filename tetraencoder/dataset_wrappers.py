import copy
from functools import partial
from typing import List, Tuple
import multiprocessing

import datasets
import numpy as np
from sentence_transformers import InputExample
from random import randrange, choice

# special tokens
Q_TOKEN = "[Q]"
S_TOKEN = "[S]"
P_TOKEN = "[P]"
O_TOKEN = "[O]"
N_TOKEN = ""
SPECIAL_TOKENS = [Q_TOKEN, S_TOKEN, P_TOKEN, O_TOKEN, N_TOKEN]

# hyperparams for hard negatives
CORRUPTION_BATCH_SIZE = 10000  # The higher the better
SYMMETRICAL_RELATIONSHIPS = [
    "taxon synonym",
    "partner in business or sport",
    "opposite of",
    "partially coincident with",
    "physically interacts with",
    "partner",
    "relative",
    "related category",
    "connects with",
    "twinned administrative body",
    "different from",
    "said to be the same as",
    "sibling",
    "adjacent station",
    "shares border with",
]

# general variable to use by default in num_procs
NCPUS = multiprocessing.cpu_count()


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


def batch_linearize_rdf(examples, rdf_key="triples", output_key="rdf_linearized"):
    examples[output_key] = [linearize_rdf(rdf) for rdf in examples[rdf_key]]
    return examples


# unitary example function for `batch_mix_triples`
def replace_random_triples(triples, replacements: Tuple[List, List, List], max_tries=1000):
    encoded_rdf = ""
    for triple in triples:
        replacement_spot = randrange(3)
        replacement = triple[replacement_spot]
        tries = 0
        while replacement == triple[replacement_spot]:
            replacement = choice(replacements[replacement_spot])
            tries += 1
            if tries == max_tries:
                raise ValueError(
                    f"We keep returning the same thing, {triple[replacement_spot]} as a "
                    f"{['subject', 'property', 'object'][replacement_spot]}. "
                    f"Is your list of replacements large enough?\n\n{replacements[replacement_spot]}")
        if len(triple) == 3:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {O_TOKEN} {triple[2]} "
        else:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {triple[2]} {O_TOKEN} {triple[3]} "
    return encoded_rdf


# batch-level function to replace random elements in graphs with elements found somewhere else in the batch
def batch_mix_triples(examples, rdf_key="triples", max_tries=1000):
    replacements = ([triple[0] for rdf in examples[rdf_key] for triple in rdf],
                    [triple[1] for rdf in examples[rdf_key] for triple in rdf],
                    [triple[2] for rdf in examples[rdf_key] for triple in rdf])
    examples["rdf_mixed"] = [replace_random_triples(rdf, replacements, max_tries=max_tries) for rdf in examples[rdf_key]]
    return examples


def _invert_triple(triple):
    output = copy.deepcopy(triple)
    output[0] = triple[-1]
    output[-1] = triple[0]
    return output


def invert_all_triples(example, rdf_key="triples"):
    # if all triples in the RDF are symmetrical relationships, return None
    if all([triple[1] in SYMMETRICAL_RELATIONSHIPS for triple in example[rdf_key]]):
        example["rdf_inverted"] = None
    else:
        example["rdf_inverted"] = linearize_rdf([_invert_triple(triple) for triple in example[rdf_key]])
    return example


def invert_one_triple(example, rdf_key="triples"):
    # if all triples in the RDF are symmetrical relationships, return None
    possible_swap_spots = [i for i in range(len(example[rdf_key])) if example[rdf_key][i][1] not in SYMMETRICAL_RELATIONSHIPS]
    if len(possible_swap_spots) == 0:
        example["rdf_inverted"] = None
    else:
        # we invert at random one of the non-symmetrical relationship triples
        replacement_spot = possible_swap_spots[randrange(len(possible_swap_spots))]
        inverted_rdf = [_invert_triple(example[rdf_key][i]) if i == replacement_spot else example[rdf_key][i] for i in range(len(example[rdf_key]))]
        example["rdf_inverted"] = linearize_rdf(inverted_rdf)
    return example


class InputExampleDataset:

    def __init__(self, previous_text_key=None, map_num_proc=None):
        self.dataset = None
        self.corruption = []
        self.previous_text_key = previous_text_key
        self.map_num_proc = map_num_proc if map_num_proc is not None else NCPUS

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )

    # dependent on the dataset, will return an InputExample of some sort. This wrapper exists to return InputExample pairs
    def __getitem__(self, item):
        return NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    # copying dataset methods for simplicity
    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, **kwargs)

    def filter(self, *args, **kwargs):
        self.dataset = self.dataset.filter(*args, **kwargs)

    def shuffle(self, seed):
        self.dataset = self.dataset.shuffle(seed)

    def select(self, *args, **kwargs):
        self.dataset = self.dataset.select(*args, **kwargs)

    def to_json(self, path):
        self.dataset.to_json(path)

    def rename_column(self, name1, name2):
        self.dataset = self.dataset.rename_column(name1, name2)

    # more practical if everyone has the same text key - triples is handled differently as there is more variation
    def uniformize_text_key(self, text_key="text"):
        if self.previous_text_key is not None and self.previous_text_key in self.dataset.column_names:
            self.rename_column(self.previous_text_key, text_key)

    # used to keep the high-scoring pairs for some similarity metric
    def filter_by_similarity(self, remaining_fraction, similarity_key="similarity"):
        assert remaining_fraction <= 1
        assert similarity_key in self.dataset.column_names, f"This doesn't have a {similarity_key} score"
        similarities = self.dataset[similarity_key]
        cutoff_index = round(remaining_fraction * len(self))
        # cutoff_index = 0 causes unexpected behaviour
        if cutoff_index == 0:
            cutoff_index = 1
        cutoff_value = np.partition(similarities, -cutoff_index)[-cutoff_index]
        self.filter(lambda x: x[similarity_key] >= cutoff_value)


class MsMarcoDataset(InputExampleDataset):
    def __init__(self, data_file, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return InputExample(texts=[Q_TOKEN + self.dataset[item]["texts"][0], self.dataset[item]["texts"][1]])


class KelmDataset(InputExampleDataset):
    def __init__(self, data_file, seed=1951, **kwargs):
        super().__init__(previous_text_key="gen_sentence", **kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(batch_linearize_rdf, batched=True, num_proc=self.map_num_proc)
        self.shuffle(seed=seed)
        self.map(partial(batch_mix_triples, rdf_key="triples"), batched=True, num_proc=self.map_num_proc,
                 batch_size=CORRUPTION_BATCH_SIZE)
        self.map(partial(invert_one_triple, rdf_key="triples"), num_proc=self.map_num_proc)
        self.uniformize_text_key()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        outputs = [example["text"], example["rdf_linearized"]]
        if "mix" in self.corruption:
            outputs.append(example["rdf_mixed"])
        if "invert" in self.corruption and example["rdf_inverted"] is not None:
            outputs.append(example["rdf_inverted"])
        return InputExample(texts=outputs)

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class GooAqDataset(InputExampleDataset):
    def __init__(self, data_file, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return InputExample(texts=[Q_TOKEN + self.dataset[item]["question"], self.dataset[item]["answer"]])

    def questions(self):
        return [Q_TOKEN + question for question in self.dataset["question"]]

    def answers(self):
        return self.dataset["answer"]


class TekgenDataset(InputExampleDataset):
    def __init__(self, data_file, seed=1951, **kwargs):
        super().__init__(previous_text_key="sentence", **kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(batch_linearize_rdf, batched=True, num_proc=self.map_num_proc)
        self.shuffle(seed=seed)
        self.map(partial(batch_mix_triples, rdf_key="triples"), batched=True, num_proc=self.map_num_proc,
                 batch_size=CORRUPTION_BATCH_SIZE)
        self.map(partial(invert_one_triple, rdf_key="triples"), num_proc=self.map_num_proc)
        self.uniformize_text_key()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        outputs = [example["text"], example["rdf_linearized"]]
        if "mix" in self.corruption:
            outputs.append(example["rdf_mixed"])
        if "invert" in self.corruption and example["rdf_inverted"] is not None:
            outputs.append(example["rdf_inverted"])
        return InputExample(texts=outputs)

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class WebNlgDataset(InputExampleDataset):
    def __init__(self, data_file, seed=1951, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True, num_proc=self.map_num_proc)
        self.map(partial(batch_mix_triples, rdf_key="triples"), batched=True, num_proc=self.map_num_proc,
                 batch_size=CORRUPTION_BATCH_SIZE)
        self.map(partial(invert_one_triple, rdf_key="triples"), num_proc=self.map_num_proc)
        self.shuffle(seed=seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        outputs = [example["text"], example["rdf_linearized"]]
        if "mix" in self.corruption:
            outputs.append(example["rdf_mixed"])
        if "invert" in self.corruption and example["rdf_inverted"] is not None:
            outputs.append(example["rdf_inverted"])
        return InputExample(texts=outputs)

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


def extract_sq_triplets(examples, src_key, rdf_key):
    examples[rdf_key] = [[src.replace("[QF]", N_TOKEN).split(" | ")] for src in examples[src_key]]
    return examples


class SQDataset(InputExampleDataset):
    def __init__(self, data_file, seed=1951, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("csv", data_files=data_file)["train"]
        self.incomplete_triple = False
        self.map(partial(extract_sq_triplets, src_key="src_prime", rdf_key="triples"), batched=True,
                 num_proc=self.map_num_proc)
        self.map(partial(extract_sq_triplets, src_key="src_prime_noqf", rdf_key="incomplete_triples"), batched=True,
                 num_proc=self.map_num_proc)
        self.map(batch_linearize_rdf, batched=True, num_proc=self.map_num_proc)
        self.map(partial(batch_linearize_rdf, rdf_key="incomplete_triples", output_key="incomplete_rdf_linearized"),
                 batched=True, num_proc=self.map_num_proc)
        self.shuffle(seed=seed)
        self.map(partial(batch_mix_triples, rdf_key="triples"), batched=True, num_proc=self.map_num_proc,
                 batch_size=CORRUPTION_BATCH_SIZE)
        self.map(partial(invert_one_triple, rdf_key="triples"), num_proc=self.map_num_proc)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        rdf_linearized = example["incomplete_rdf_linearized"] if self.incomplete_triple else example["rdf_linearized"]
        question = example["refs"]

        outputs = [question, rdf_linearized]
        if "mix" in self.corruption:
            outputs.append(example["rdf_mixed"])
        if "invert" in self.corruption and example["rdf_inverted"] is not None:
            outputs.append(example["rdf_inverted"])
        return InputExample(texts=outputs)

    def switch_to_incomplete_triples(self):
        self.incomplete_triple = True

    def switch_to_complete_triples(self):
        self.incomplete_triple = False

    def rdfs(self, incomplete=None):
        incomplete = self.incomplete_triple if incomplete is None else incomplete
        if incomplete:
            return self.dataset["incomplete_rdf_linearized"]
        else:
            return self.dataset["rdf_linearized"]

    def questions(self):
        return [refs[0] for refs in self.dataset["refs"]]


class GenWikiDataset(InputExampleDataset):
    def __init__(self, data_file, seed=1951, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(GenWikiDataset.batched_fill_in_entities, batched=True, num_proc=self.map_num_proc)
        self.rename_column("text", "unfilled_text")
        self.rename_column("filled_text", "text")
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True, num_proc=self.map_num_proc)
        self.shuffle(seed=seed)
        self.map(partial(batch_mix_triples, rdf_key="triples"), batched=True, num_proc=self.map_num_proc,
                 batch_size=CORRUPTION_BATCH_SIZE)
        self.map(partial(invert_one_triple, rdf_key="triples"), num_proc=self.map_num_proc)

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def fill_in_entities(cls, text, entities):
        for i, entity in enumerate(entities):
            text = text.replace(f"<ENT_{i}>", entity)
        return text

    @classmethod
    def batched_fill_in_entities(cls, examples):
        examples["filled_text"] = [cls.fill_in_entities(text, entities) for text, entities in
                                   zip(examples["text"], examples["entities"])]
        return examples

    def __getitem__(self, item):
        example = self.dataset[item]
        outputs = [example["text"], example["rdf_linearized"]]
        if "mix" in self.corruption:
            outputs.append(example["rdf_mixed"])
        if "invert" in self.corruption and example["rdf_inverted"] is not None:
            outputs.append(example["rdf_inverted"])
        return InputExample(texts=outputs)

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class TRexDataset(InputExampleDataset):
    def __init__(self, data_file, seed=1951, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True, num_proc=self.map_num_proc)
        self.shuffle(seed=seed)
        self.map(partial(batch_mix_triples, rdf_key="triples"), batched=True, num_proc=self.map_num_proc,
                 batch_size=CORRUPTION_BATCH_SIZE)
        self.map(partial(invert_one_triple, rdf_key="triples"), num_proc=self.map_num_proc)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        outputs = [example["text"], example["rdf_linearized"]]
        if "mix" in self.corruption:
            outputs.append(example["rdf_mixed"])
        if "invert" in self.corruption and example["rdf_inverted"] is not None:
            outputs.append(example["rdf_inverted"])
        return InputExample(texts=outputs)

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class MPWWDataset(InputExampleDataset):
    def __init__(self, data_file, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True, num_proc=self.map_num_proc)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        return InputExample(texts=[example["text"], example["rdf_linearized"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


# train_dataset_builders
dataset_builders = {"msmarco": MsMarcoDataset, "kelm": KelmDataset, "gooaq": GooAqDataset, "tekgen": TekgenDataset,
                    "trex": TRexDataset}