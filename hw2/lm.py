#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
import pdb

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        vocab_set = set(self.vocab())
        words_set  = set([w for s in corpus for w in s])
        numOOV = len(words_set - vocab_set)
        if False:
            # for w in words_set:
            #     if not w in vocab_set:
            #         print(w)
            pdb.set_trace()
        return pow(2.0, self.entropy(corpus, numOOV))

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass

    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass

    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass

    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        """Count the number of appearance of each word (macro word matrix)"""
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        """Call inc_word() for a sentence and add 'EOS' to the macro word matrix.
        Remarkably, there is no 'eos' in the original sentences.
        """
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs. (Strange definition of 'normalize')"""
        # total number of words in the corpus
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        # denominator for calculation of the probalility
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

class Bigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        # self.vocabulary = []
        self.lunk_prob = log(unk_prob, 2)
        self.EOS_as_start_prob = 0.0001

    def inc_word_mat(self, token):
        """Count the number of appearance of each word (macro word matrix)"""
        if token[0] in self.model:
            if token[1] in self.model[token[0]]:
                self.model[token[0]][token[1]] += 1.0
            else:
                self.model[token[0]][token[1]] = 1.0
        else:
            self.model[token[0]] = {token[1]: 1.0}

    def fit_sentence(self, sentence):
        """Call inc_word() for a sentence and add 'EOS' to the macro word matrix
        Loop through all the possible combination of the bigram"""
        for token in self.combination_gen(sentence, comb=2):
            self.inc_word_mat(token)

    def norm(self):
        """Normalize and convert to log2-probs. (Strange definition of 'normalize')""" 
        # add EOS to the row-keys
        v = self.vocab()
        for word in v:
            if 'END_OF_SENTENCE' in self.model.keys():
                self.model['END_OF_SENTENCE'][word] = self.EOS_as_start_prob
            else:
                self.model['END_OF_SENTENCE'] = {word: self.EOS_as_start_prob}

        for word in self.model:
            tot = self.denominator(self.model, word)
            ltot = log(tot, 2)
            for key in self.model[word].keys():
                self.model[word][key] = log(self.model[word][key], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            if previous == []:
                previous = ['START_OF_SENTENCE']
            return self.model[previous[-1]][word]
        else:
            pdb.set_trace()
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

    def combination_gen(self, sentence, comb=2):
        """Generate all possible combination in a sentence with the length of combination"""
        output = []
        for i in range(-1 , len(sentence) + 2 - comb): # [len(sentence + 2) + 2 - comb + 1] iterations
            tup = []
            if i < 0:
                tup.append('START_OF_SENTENCE')
                for j in range(1, comb, 1):
                    if i+j < len(sentence):
                        tup.append(sentence[i+j])
                    else: # There can't be two steps exceeds the length of the sentence
                        tup.append('END_OF_SENTENCE')
            else:
                for j in range(comb):
                    if i+j < len(sentence):
                        tup.append(sentence[i+j])
                    else:
                        tup.append('END_OF_SENTENCE')
            output.append(tuple(tup))
        return output

    def denominator(self, model, word):
        """Calcualte the total number of the apperance of a denominator word in conditional prob"""
        total = 0.0
        for key in model[word].keys():
            total += model[word][key]
        return total