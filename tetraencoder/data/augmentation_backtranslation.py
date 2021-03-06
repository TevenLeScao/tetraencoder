# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from nltk.translate import gleu_score
from unicodedata import normalize
from datasets import load_dataset
import pickle
from tqdm import tqdm
import numpy as np
import torch


class MultilingualBackTranslation:
    supported_languages = ['af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'cy',
                           'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu',
                           'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka',
                           'kk', 'km', 'kn', 'ko', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr',
                           'ms', 'my', 'ne', 'nl', 'no', 'ns', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd',
                           'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th', 'tl', 'tn', 'tr',
                           'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu']

    def __init__(self, src_lang: str = 'en', pivot_lang: str = 'zh', device='cuda:0'):

        # Validate and set src_lang
        if src_lang not in self.supported_languages:
            raise ValueError('{} is not a supported src_lang'.format(src_lang))
        self.src_lang = src_lang

        # Validate and set pivot_lang
        if pivot_lang not in self.supported_languages:
            raise ValueError('{} is not a supported pivot_lang'.format(pivot_lang))
        self.pivot_lang = pivot_lang

        # Initialize model and tokenizer
        self.device = torch.device(device)
        self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M').to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

    def _translate_(self, sentences: list, src_lang: str, target_lang: str) -> list:

        # Set src_lang
        self.tokenizer.src_lang = src_lang

        # Tokenize sentences
        encoded_source_sentence = self.tokenizer(sentences, padding=True, return_tensors="pt")
        encoded_source_sentence = encoded_source_sentence.to(self.device)
        # Get translation
        generated_target_tokens = self.model.generate(**encoded_source_sentence,
                                                      forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))

        # Rebuild translated sentences
        target_sentence = self.tokenizer.batch_decode(generated_target_tokens, skip_special_tokens=True)

        return target_sentence

    def generate(self, sentences: list) -> list:

        # Translate from Source to Pivot
        pivot_sentences = self._translate_(sentences, self.src_lang, self.pivot_lang)

        # Backtranslate from Pivot to Source
        if self.pivot_lang != self.src_lang:
            source_sentences = self._translate_(pivot_sentences, self.pivot_lang, self.src_lang)
        else:
            source_sentences = pivot_sentences

        return source_sentences


"""# ParEval"""

SENTENCE_EMBEDDEER = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')


def process_text(text: str) -> str:
    """Basic string processing
    
    Lowercase, normalize, and remove non alphanumeric characters from a string.
    This is done so the surface similarity score does not changed because if minor things.    
    
    Parameters
    ----------
    text : str
        The text to be processed.
    
    Returns
    -------
    str
        a string with the processed text
    
    Raises
    ------
    TypeError
        If text is not of type str.
    """

    # Sanity Test
    if not isinstance(text, str):
        raise TypeError('text should be of type str, found {}'.format(type(text)))

    # Remove none alphanumeric characters
    normalized_text = ' '.join(''.join(c for c in word if c.isalnum()) for word in text.split())

    # Normalize Characters
    normalized_text = normalize('NFKC', normalized_text)

    # Lowercase
    normalized_text = normalized_text.lower()

    # Return simple string
    return normalized_text


def pareval_pair(text1: str, text2: str) -> dict:
    """Generate the pareval score of a pair of texts.
    
    Parameters
    ----------
    text1 : str
        The first text to evaluate.
    text2 : str
        The second text to evaluate.
    
    Returns
    -------
    dict
        a dictionary with pareval_score, semantic_similarity, surface_similarity, and settings.
    
    Raises
    ------
    TypeError
        If text1 is not of type str.
    TypeError
        If text2 is not of type str.
    """

    # Sanity test
    if not isinstance(text1, str):
        raise TypeError('text1 should be of type str, found {}'.format(type(text1)))

    if not isinstance(text2, str):
        raise TypeError('text2 should be of type str, found {}'.format(type(text2)))

    # Get the two texts pareval score
    pareval_summary = pareval_batch([text1], [text2])

    # Return the output
    return pareval_summary


def pareval_batch(text1_batch: list, text2_batch: list, return_type: str = 'score') -> dict:
    """Generate the pareval score of a batch of pairs of texts.
    
    Parameters
    ----------
    text1_batch : list
        A list with all the first texts to evaluate.
    text2_batch : list
        A list with all the second texts to evaluate.
    
    Returns
    -------
    dict
        a dictionary with pareval_score, semantic_similarity, surface_similarity, and settings.
    
    Raises
    ------
    TypeError
        If text1_batch is not of type list.
    TypeError
        If text2_batch is not of type list.
    ValueError
        If text1_batch and text2_batch have a different number of elements.
    TypeError
        If text1 has elements that are not of type str.
    TypeError
        If text2has elements that are not of type str.
    """

    # Sanity test
    if not isinstance(text1_batch, list):
        raise TypeError('text1_batch should be of type list, found {}'.format(type(text1_batch)))

    if not isinstance(text2_batch, list):
        raise TypeError('text2_batch should be of type list, found {}'.format(type(text2_batch)))

    len_text1_batch = len(text1_batch)
    len_text2_batch = len(text2_batch)
    if len_text1_batch != len_text2_batch:
        raise ValueError('text1_batch and text2_batch are of different sizes, found {} and {}'.format(len_text1_batch,
                                                                                                      len_text2_batch))

    for position, text in enumerate(text1_batch):
        if not isinstance(text, str):
            raise TypeError(
                'All elements of text1_batch should be of type str, found {} at position {}'.format(type(text),
                                                                                                    position))

    for position, text in enumerate(text2_batch):
        if not isinstance(text, str):
            raise TypeError(
                'All elements of text2_batch should be of type str, found {} at position {}'.format(type(text),
                                                                                                    position))

    # Set up the output
    pareval_summary = {'pareval_score': [], 'semantic_similarity': [], 'surface_similarity': [],
                       'settings': 'paraphrase-xlm-r-multilingual-v1+Google-BLEU'}

    # Generate sentence embeddings of all the texts
    text1_embeddings = SENTENCE_EMBEDDEER.encode(text1_batch)
    text2_embeddings = SENTENCE_EMBEDDEER.encode(text2_batch)

    # Generate the ParEval score for each pair of texts
    for i in range(len_text1_batch):
        # Calculate semantic similarity (Embeddings cosine similarity)
        semantic_similarity = util.pytorch_cos_sim([text1_embeddings[i]], [text2_embeddings[i]])[0][0].item()

        # Preprocess text
        processed_text_1 = process_text(text1_batch[i])
        processed_text_2 = process_text(text2_batch[i])

        # Tokenize text
        tokenized_text_1 = SENTENCE_EMBEDDEER.tokenizer.tokenize(processed_text_1)
        tokenized_text_2 = SENTENCE_EMBEDDEER.tokenizer.tokenize(processed_text_2)

        # Calculate surface similarity (Google-BLEU)
        surface_similarity = gleu_score.sentence_gleu([tokenized_text_1], tokenized_text_2)

        # Calculate the surface similarity penalty
        surface_similarity_penalty = ((surface_similarity ** 2) * semantic_similarity) / 2

        # Calculate the ParEval score
        pareval_score = semantic_similarity - surface_similarity_penalty

        # Format the output
        pareval_summary['pareval_score'].append(pareval_score)
        pareval_summary['semantic_similarity'].append(semantic_similarity)
        pareval_summary['surface_similarity'].append(surface_similarity)

    # Return the output
    return pareval_summary


"""# Misc"""


def get_chunk_ranges(a, n):
    current = 0
    ranges = []
    while current + n <= a:
        ranges.append(range(current, current + n))
        current += n
    ranges.append(range(current, a))
    return ranges


def get_grouping(chunk):
    return [len(qs) for qs in chunk]


def flatten_chunk(chunk):
    return [q for qs in chunk for q in qs]


def regroup_chunk(flat_chunk, grouping):
    chunk = []
    current = 0
    for g in grouping:
        chunk.append(flat_chunk[current: current + g])
        current += g
    return chunk


"""# Zeroshot Question"""

kilt_zeroshot_re = load_dataset("kilt_tasks", name="structured_zeroshot")
# clear_output()

qn_templates = {}
for split in ['train', 'validation', 'test']:
    qn_templates[split] = kilt_zeroshot_re[split].map(
        lambda x: {'id': x['id'], 'question': x['meta']['template_questions']})
    # clear_output()

parser = ArgumentParser()
parser.add_argument('-l', '--language', type=str, help='language to process', default="de")
parser.add_argument('-d', '--device', type=int, help='device to use', default=0)
parser.add_argument('--cpu', action="store_true")
parser.add_argument('-b', '--batch-size', type=int, help='device to use', default=16)
# recommended choices: de, fr, pl
args = parser.parse_args()
lang = args.language

subsets = ['train', 'validation', 'test']

paraphrases = {}
xpareval = {lang: {}}
paraphrases[lang] = {}
xpareval[lang]["base"] = {"pareval": [], "semantic": [], "surface": []}
xpareval[lang]["average"] = {"pareval": 0, "semantic": 0, "surface": 0}
for split in subsets:
    paraphrases[lang][split] = []

batch_size = args.batch_size

print('Language: {}'.format(lang))
backtranslator = MultilingualBackTranslation('en', lang, device="cpu" if args.cpu else f"cuda:{args.device}")
for split in subsets:
    print('Split: {}'.format(split))
    total_samples = len(qn_templates[split])
    chunks = get_chunk_ranges(total_samples, batch_size)
    total_chunks = len(chunks)
    print(f"total_chunks: {total_chunks}")
    for c, chunk in tqdm(enumerate(chunks)):
        if c * batch_size >= len(paraphrases[lang][split]):
            raw_questions = qn_templates['train'].select(chunk)['question']
            grouping = get_grouping(raw_questions)
            questions = flatten_chunk(raw_questions)
            backtranslated_questions = backtranslator.generate(questions)
            backtranslated_groups = regroup_chunk(backtranslated_questions, grouping)
            parallel_list = [{"originals": raw_group, "backtranslated": backtranslated_group} for
                             raw_group, backtranslated_group in zip(raw_questions, backtranslated_groups)]
            pareval_summary = pareval_batch(questions, backtranslated_questions)
            paraphrases[lang][split].extend(parallel_list)
            xpareval[lang]["base"]["pareval"].extend(pareval_summary['pareval_score'])
            xpareval[lang]["base"]["semantic"].extend(pareval_summary['semantic_similarity'])
            xpareval[lang]["base"]["surface"].extend(pareval_summary['surface_similarity'])
        if c % 500:
            for metric in xpareval[lang]["base"]:
                xpareval[lang]["average"][f"{metric}"] = sum(xpareval[lang]["base"][metric]) / len(
                    xpareval[lang]["base"][metric])
            with open(f'backtranslated_{lang}.bin', 'wb') as file:
                pickle.dump([paraphrases, xpareval], file, protocol=4)
for metric in xpareval[lang]["base"]:
    xpareval[lang]["average"][f"{metric}"] = sum(xpareval[lang]["base"][metric]) / len(xpareval[lang]["base"][metric])
with open(f'backtranslated_{lang}.bin', 'wb') as file:
    pickle.dump([paraphrases, xpareval], file, protocol=4)
print()
