from functools import partial
from typing import List, Tuple

import datasets
from sentence_transformers import InputExample
from random import randrange, choice

Q_TOKEN = "[Q]"
S_TOKEN = "[S]"
P_TOKEN = "[P]"
O_TOKEN = "[O]"
N_TOKEN = "[N]"
SPECIAL_TOKENS = [Q_TOKEN, S_TOKEN, P_TOKEN, O_TOKEN, N_TOKEN]
CORRUPTION_BATCH_SIZE = 10000 # The higher the better


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


def corrupt_rdf(triples, replacements: Tuple[List, List, List]):

    encoded_rdf = ""
    for triple in triples:
        replacement_spot = randrange(3)
        replacement = triple[replacement_spot]
        tries = 0
        while replacement == triple[replacement_spot]:
            replacement = choice(replacements[replacement_spot])
            tries += 1
            if tries == 10:
                raise ValueError("We keep returning the same thing. Is your list of replacements large enough?")
        if len(triple) == 3:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {O_TOKEN} {triple[2]} "
        else:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {triple[2]} {O_TOKEN} {triple[3]} "
    return encoded_rdf


def batch_corrupt_rdf(examples, rdf_key):
    replacements = ([triple[0] for rdf in examples[rdf_key] for triple in rdf],
                    [triple[1] for rdf in examples[rdf_key] for triple in rdf],
                    [triple[2] for rdf in examples[rdf_key] for triple in rdf])
    examples["rdf_corrupted"] = [corrupt_rdf(rdf, replacements) for rdf in examples[rdf_key]]
    return examples


class InputExampleDataset:

    def __init__(self):
        self.dataset = None
        self.corruption = False

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )

    def __getitem__(self, item):
        return NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, **kwargs)

    def shuffle(self, seed):
        self.dataset = self.dataset.shuffle(seed)

    def select(self, *args, **kwargs):
        self.dataset = self.dataset.select(*args, **kwargs)

    def to_json(self, path):
        self.dataset.to_json(path)


class MsMarcoDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return InputExample(texts=[Q_TOKEN + self.dataset[item]["texts"][0], self.dataset[item]["texts"][1]])


class KelmDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(batch_linearize_rdf, batched=True)
        self.map(partial(batch_corrupt_rdf, rdf_key="triples"), batched=True, batch_size=CORRUPTION_BATCH_SIZE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        if self.corruption:
            return InputExample(texts=[example["gen_sentence"], example["rdf_linearized"], example["rdf_corrupted"]])
        else:
            return InputExample(texts=[example["gen_sentence"], example["rdf_linearized"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class GooAqDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
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
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(batch_linearize_rdf, batched=True)
        self.map(partial(batch_corrupt_rdf, rdf_key="triples"), batched=True, batch_size=CORRUPTION_BATCH_SIZE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        if self.corruption:
            return InputExample(texts=[example["sentence"], example["rdf_linearized"], example["rdf_corrupted"]])
        else:
            return InputExample(texts=[example["rdf_linearized"], example["sentence"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class WebNlgWikidataDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True)
        self.map(partial(batch_corrupt_rdf, rdf_key="triples"), batched=True, batch_size=CORRUPTION_BATCH_SIZE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        if self.corruption:
            return InputExample(texts=[example["sentence"], example["rdf_linearized"], example["rdf_corrupted"]])
        else:
            return InputExample(texts=[example["sentence"], example["rdf_linearized"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


def extract_sq_triplets(examples, src_key, rdf_key):
    examples[rdf_key] = [[src.replace("[QF]", N_TOKEN).split(" | ")] for src in examples[src_key]]
    return examples


class SQDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("csv", data_files=data_file)["train"]
        self.incomplete_triple = False
        self.map(partial(extract_sq_triplets, src_key="src_prime", rdf_key="triples"), batched=True)
        self.map(partial(extract_sq_triplets, src_key="src_prime_noqf", rdf_key="incomplete_triples"), batched=True)
        self.map(batch_linearize_rdf, batched=True)
        self.map(partial(batch_linearize_rdf, rdf_key="incomplete_triples", output_key="incomplete_rdf_linearized"), batched=True)
        self.map(partial(batch_corrupt_rdf, rdf_key="triples"), batched=True, batch_size=CORRUPTION_BATCH_SIZE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        rdf_linearized = example["incomplete_rdf_linearized"] if self.incomplete_triple else example["rdf_linearized"]
        question = example["refs"][0]

        if self.corruption:
            return InputExample(texts=[question, rdf_linearized, example["rdf_corrupted"]])
        else:
            return InputExample(texts=[question, rdf_linearized])

    def switch_to_incomplete_triples(self):
        self.incomplete_triple = True

    def switch_to_complete_triples(self):
        self.incomplete_triple = False

    def rdfs(self):
        if self.incomplete_triple:
            return [linearize_rdf([triple.replace("QF", N_TOKEN)]) for triple in self.dataset["src_prime_noqf"]]
        else:
            return self.dataset["rdf_linearized"]

    def questions(self):
        return [refs[0] for refs in self.dataset["refs"]]


class GenWikiDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(GenWikiDataset.batched_fill_in_entities, batched=True)
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True)
        self.map(partial(batch_corrupt_rdf, rdf_key="triples"), batched=True, batch_size=CORRUPTION_BATCH_SIZE)

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def fill_in_entities(cls, text, entities):
        for i, entity in enumerate(entities):
            text = text.replace(f"<ENT_{i}>", entity)
        return text

    @classmethod
    def batched_fill_in_entities(cls, examples):
        examples["filled_text"] = [cls.fill_in_entities(text, entities) for text, entities in zip(examples["text"], examples["entities"])]
        return examples

    def __getitem__(self, item):
        example = self.dataset[item]
        if self.corruption:
            return InputExample(texts=[example["filled_text"], example["linearized_rdf"], example["corrupted_rdf"]])
        else:
            return InputExample(texts=[example["filled_text"], example["linearized_rdf"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["filled_text"]


class TRexDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True)
        self.map(partial(batch_corrupt_rdf, rdf_key="triples"), batched=True, batch_size=CORRUPTION_BATCH_SIZE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        return InputExample(texts=[example["rdf_linearized"], example["text"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]
