from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple
from time import time
import sys 


import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.
        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.
        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices
        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result
        


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters


    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        posteriors = []

        for i in range(len(parallel_corpus)):

            translate = self.translation_probs[np.ix_(
                parallel_corpus[i].source_tokens, 
                parallel_corpus[i].target_tokens)]

            posteriors.append(translate / np.sum(translate, axis=0))

        return posteriors


    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        self.translation_probs *= 0

        for i in range(len(parallel_corpus)):

            np.add.at(self.translation_probs, np.ix_(
                parallel_corpus[i].source_tokens, 
                parallel_corpus[i].target_tokens), posteriors[i])


        self.translation_probs /= np.sum(self.translation_probs, axis=1).reshape(self.num_source_words, 1)


        return self._compute_elbo(parallel_corpus, posteriors)


    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        elbo = 0
    
        for i in range(len(parallel_corpus)):
            len_s = len(parallel_corpus[i].source_tokens)
            len_t = len(parallel_corpus[i].target_tokens)

            elbo += np.sum(posteriors[i] * np.log(self.translation_probs[np.ix_(
                parallel_corpus[i].source_tokens, 
                parallel_corpus[i].target_tokens)]) 
                    - posteriors[i] * np.log(len_s) 
                    - posteriors[i] * np.log(posteriors[i]))
            
        return elbo


    def fit(self, parallel_corpus):
        history = []
        eps = 1e-7

        start_t = time()

        for i in range(self.num_iters):
            print("Iter ", i, ", Time elapsed %.3f" %(time() - start_t), ';', sep='')
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)

            if len(history) > 1 and history[-1] < history[-2] - eps:
                break

        return history


    def align(self, sentences):

        all_aligns = []
        
        for pair in sentences:
            source = pair.source_tokens
            target = pair.target_tokens
            len_s = len(source)
            len_t = len(target)

            cur_aligns = []

            for t_ind in range(len_t):
                t = target[t_ind]
                max_prob = (0, -1)

                for s_ind in range(len_s):
                    s = source[s_ind]

                    max_prob = max(max_prob, (self.translation_probs[s][t], s_ind))

                cur_aligns.append((max_prob[1] + 1, t_ind + 1))

            all_aligns.append(cur_aligns)

        return all_aligns



class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.
        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence
        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        pass

    def _e_step(self, parallel_corpus):
        pass

    def _compute_elbo(self, parallel_corpus, posteriors):
        pass

    def _m_step(self, parallel_corpus, posteriors):
        pass


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.
        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence
        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        pass

    def _e_step(self, parallel_corpus):
        pass

    def _compute_elbo(self, parallel_corpus, posteriors):
        pass

    def _m_step(self, parallel_corpus, posteriors):
        pass











# a slower but easier to understand version




class SlowerAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters


    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        posteriors = []

        for i in range(len(parallel_corpus)):

            source = parallel_corpus[i].source_tokens
            target = parallel_corpus[i].target_tokens
            len_s = len(source)
            len_t = len(target)


            translate = self.translation_probs[np.ix_(source, target)]

            probs = translate.sum(axis=1)
            counts = translate / probs.reshape(len(probs), 1)


            posteriors.append(counts)

        return posteriors



    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.
        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).
        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs = np.zeros(self.translation_probs.shape)
        target_probs = np.zeros((self.num_target_words))

        for i in range(len(parallel_corpus)):
            source = parallel_corpus[i].source_tokens
            target = parallel_corpus[i].target_tokens
            len_s = len(source)
            len_t = len(target)

            target_probs[target] += posteriors[i].sum(axis=0)
            np.add.at(self.translation_probs, np.ix_(source, target), posteriors[i])

        #print(translation_probs_new.sum(axis=1)[0])

        self.translation_probs /= target_probs.reshape(1, self.num_target_words)

        return self._compute_elbo(parallel_corpus, posteriors)


    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        elbo = 0
    
        for i in range(len(parallel_corpus)):
            source = parallel_corpus[i].source_tokens
            target = parallel_corpus[i].target_tokens

            len_s = len(source)
            len_t = len(target)

            translate = self.translation_probs[np.ix_(source, target)]

            qlogq = posteriors[i] * np.log(posteriors[i])
            prob = posteriors[i] * np.log(translate)
            scale = posteriors[i] * np.log(len_s)
            #print(qlogq.shape)
            #print(prob.shape)
            #print(scale.shape)

            elbo += np.sum(prob - scale - qlogq)
            
        return elbo


    def fit(self, parallel_corpus):
        start_t = time()

        history = []
        eps = 1e-7

        for i in range(self.num_iters):
            print("Iter ", i, ", Time elapsed %.3f" %(time() - start_t), ';', sep='')
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)

            if len(history) > 1 and history[-1] < history[-2] - eps:
                break

        return history


    def align(self, sentences):

        all_aligns = []
        
        for pair in sentences:
            source = pair.source_tokens
            target = pair.target_tokens
            len_s = len(source)
            len_t = len(target)

            cur_aligns = []

            for s_ind in range(len_s):
                s = source[s_ind]
                max_prob = (0, -1)

                for t_ind in range(len_t):
                    t = target[t_ind]

                    max_prob = max(max_prob, (self.translation_probs[s][t], t_ind))

                cur_aligns.append((s_ind + 1, max_prob[1] + 1))

            all_aligns.append(cur_aligns)

        return all_aligns
