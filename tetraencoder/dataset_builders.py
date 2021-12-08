from functools import partial

import datasets
from sentence_transformers import InputExample

Q_TOKEN = "[Q]"
S_TOKEN = "[S]"
P_TOKEN = "[P]"
O_TOKEN = "[O]"
N_TOKEN = "[N]"
SPECIAL_TOKENS = [Q_TOKEN, S_TOKEN, P_TOKEN, O_TOKEN, N_TOKEN]


def linearize_rdf(triples):
    encoded_rdf = ""
    for triple in triples:
        if len(triple) == 3:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {O_TOKEN} {triple[2]} "
        else:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {triple[2]} {O_TOKEN} {triple[3]} "
    return encoded_rdf


def batch_linearize_rdf(examples, rdf_key):
    examples["rdf_linearized"] = [linearize_rdf(rdf) if isinstance(rdf[0], list) else linearize_rdf([rdf]) for rdf in examples[rdf_key]]
    return examples


class InputExampleDataset:

    def __init__(self):
        self.dataset = None

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
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        return InputExample(texts=[example["rdf_linearized"], example["gen_sentence"]])


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
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        return InputExample(texts=[example["rdf_linearized"], example["sentence"]])


class WebNlgWikidataDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triple"), batched=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        return InputExample(texts=[example["rdf_linearized"], example["sentence"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]


class SQDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("csv", data_files=data_file)["train"]
        self.incomplete_triple = False
        self.map(partial(batch_linearize_rdf, rdf_key="src_prime"), batched=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        if self.incomplete_triple:
            rdf_linearized = linearize_rdf([example["src_prime_noqf"].replace("QF", N_TOKEN)])
        else:
            rdf_linearized = example["rdf_linearized"]
        text = example["refs"][0]
        return InputExample(texts=[rdf_linearized, text])

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
        self.map(partial(batch_linearize_rdf, rdf_key="graph"), batched=True)

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
        triples = example["graph"]
        text = self.fill_in_entities(example["text"], example["entities"])
        return InputExample(texts=[linearize_rdf(triples), text])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return [self.fill_in_entities(text, entities) for text, entities in
                zip(self.dataset["text"], self.dataset["entities"])]


class TRexDataset(InputExampleDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]
        self.map(partial(batch_linearize_rdf, rdf_key="triples"), batched=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        return InputExample(texts=[example["rdf_linearized"], example["text"]])

    def rdfs(self):
        return self.dataset["rdf_linearized"]

    def sentences(self):
        return self.dataset["text"]
