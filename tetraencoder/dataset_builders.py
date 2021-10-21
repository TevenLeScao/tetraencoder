import datasets
from sentence_transformers import InputExample

Q_TOKEN = "[Q]"
S_TOKEN = "[S]"
P_TOKEN = "[P]"
O_TOKEN = "[O]"
SPECIAL_TOKENS = [Q_TOKEN, S_TOKEN, P_TOKEN, O_TOKEN]


def linearize_rdf(triples):
    encoded_rdf = ""
    for triple in triples:
        if len(triple) == 3:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {O_TOKEN} {triple[2]}"
        else:
            encoded_rdf += f"{S_TOKEN} {triple[0]} {P_TOKEN} {triple[1]} {triple[2]} {O_TOKEN} {triple[3]}"
    return encoded_rdf


class MsMarcoDataset():
    def __init__(self, data_file):
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return InputExample(texts=[Q_TOKEN + self.dataset[item]["texts"][0], self.dataset[item]["texts"][1]])

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )


class KelmDataset():
    def __init__(self, data_file):
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        triples = example["triples"]
        text = example["gen_sentence"]
        return InputExample(texts=[linearize_rdf(triples), text])

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )


class GooAqDataset():
    def __init__(self, data_file):
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return InputExample(texts=[Q_TOKEN + self.dataset[item]["question"], self.dataset[item]["answer"]])

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )


class TekgenDataset():
    def __init__(self, data_file):
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        triples = example["triples"]
        text = example["sentence"]
        return InputExample(texts=[linearize_rdf(triples), text])

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )


class WebNlgWikidataDataset():
    def __init__(self, data_file):
        self.data_file = data_file
        self.dataset = datasets.load_dataset("json", data_files=data_file)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        triples = [example["triple"]]
        text = example["text"]
        return InputExample(texts=[linearize_rdf(triples), text])

    def rdfs(self):
        return [linearize_rdf([triple]) for triple in self.dataset["triple"]]

    def sentences(self):
        return self.dataset["text"]

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.__getitem__(
                index,
            )
