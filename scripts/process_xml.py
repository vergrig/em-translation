from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.
    Args:
        filename: Name of the file containing XML markup for labeled alignments
    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    file = open(filename, 'r', encoding='utf-8')
    file = file.read().replace('&','&amp;')

    root = ET.fromstring(file)

    sentence_pairs = []
    alignments = []

    for s in root:
        cur_eng, cur_cz, cur_sure, cur_psb = [], [], [], []

        for elem in s:
            if elem.tag == 'english' and type(elem.text) == str:
                cur_eng = elem.text.split(' ')
            if elem.tag == 'czech' and type(elem.text) == str:
                cur_cz = elem.text.split(' ')
            if elem.tag == 'sure' and type(elem.text) == str:
                txt = elem.text.split(' ')
                cur_sure = [tuple([int(y) for y in x.split('-')]) for x in txt]
            if elem.tag == 'possible' and type(elem.text) == str:
                txt = elem.text.split(' ')
                cur_psb = [tuple([int(y) for y in x.split('-')]) for x in txt]

        cur_pair = SentencePair(cur_eng, cur_cz)
        #print(cur_pair)
        sentence_pairs.append(cur_pair)

        cur_alignment = LabeledAlignment(cur_sure, cur_psb)
        alignments.append(cur_alignment)

    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.
    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language
    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
    """
    freq_source = dict()
    freq_target = dict()

    for s_pair in sentence_pairs:
        s_source = s_pair.source
        s_target = s_pair.target

        for word in s_source:
            freq_source[word] = freq_source.get(word, 0) - 1

        for word in s_target:
            freq_target[word] = freq_target.get(word, 0) - 1

    source_list = sorted(freq_source, key=freq_source.get)
    target_list = sorted(freq_target, key=freq_target.get)

    if freq_cutoff != None and len(source_list) > freq_cutoff:
        source_list = source_list[:freq_cutoff]

    if freq_cutoff != None and len(target_list) > freq_cutoff:
        target_list = target_list[:freq_cutoff]

    source_dict = {k: v for v, k in enumerate(source_list)}
    target_dict = {k: v for v, k in enumerate(target_list)}


    return (source_dict, target_dict)





def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language
    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for s_pair in sentence_pairs:
        s_source = s_pair.source
        s_target = s_pair.target

        token_source = np.array(list(map(source_dict.get, s_source)))
        token_target = np.array(list(map(target_dict.get, s_target)))

        token_source = token_source[token_source != np.array(None)]
        token_target = token_target[token_target != np.array(None)]

        if len(token_source) == 0 or len(token_target) == 0:
            continue

        pair = TokenizedSentencePair(token_source, token_target)
        tokenized_sentence_pairs.append(pair)

    return tokenized_sentence_pairs

#sentence_pairs, alignments = extract_sentences('input.xml')
#sd, td = get_token_to_index(sentence_pairs, freq_cutoff=100)
#print(sd.keys())
#sp = tokenize_sents(sentence_pairs, sd, td)
#print(sp)
