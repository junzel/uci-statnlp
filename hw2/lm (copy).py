#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
import pdb
import time

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
        numOOV = len(words_set - vocab_set) # out-of-vocabulary
        
        return pow(2.0, self.entropy(corpus, numOOV))

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            # try:
            #     if numOOV < 20065:
            #         print(self.logprob_sentence(s, numOOV))
            # except:
            #     print(numOOV)
            #     pdb.set_trace()
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
    def __init__(self, unk_prob=0.000001): # unk_prob=0.0001
        self.model = dict()
        self.unigram_model = dict()
        self.lunk_prob = log(unk_prob, 2)
        self.very_small_value = 0.0001

    def inc_word_mat(self, token):
        """Count the number of appearance of each word (macro word matrix)"""
        if token in self.model:
            self.model[token] += 1.0
        else:
            self.model[token] = 1.0

    def inc_word(self, w):
        """Count the number of appearance of each word (macro word matrix)"""
        if w in self.unigram_model:
            self.unigram_model[w] += 1.0
        else:
            self.unigram_model[w] = 1.0

    def fit_sentence(self, sentence):
        """Call inc_word() for a sentence and add 'EOS' to the macro word matrix
        Loop through all the possible combination of the bigram"""
        self.inc_word('START_OF_SENTENCE')
        # self.inc_word_mat(('START_OF_SENTENCE', 'START_OF_SENTENCE'))
        for token in self.combination_gen(sentence, comb=2):
            self.inc_word_mat(token)
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs. (Strange definition of 'normalize')""" 
        # # add EOS to the row-keys
        # v = list(self.vocab())
        # for word in v:
        #     if 'END_OF_SENTENCE' in self.model.keys():
        #         self.model['END_OF_SENTENCE'][word] = self.very_small_value
        #     else:
        #         self.model['END_OF_SENTENCE'] = {word: self.very_small_value}

        for key in self.model.keys():
            denominator = self.unigram_model[key[0]]
            l_denom = log(denominator, 2)
            self.model[key] = log(self.model[key], 2) - l_denom

    def cond_logprob(self, word, previous, numOOV):
        # assert numOOV != 0, "numOOV == 0!"
        # if numOOV == 0:
        #     pdb.set_trace()
        if previous == []:
            previous = ['START_OF_SENTENCE']
        if (previous[-1], word) in self.model:
            return self.model[(previous[-1], word)]
        else:
            try:
                return self.lunk_prob-log(numOOV, 2)
            except:
                pdb.set_trace()
                # return self.lunk_prob-log(1, 2)

    def vocab(self):
        return set(self.unigram_model.keys()) - set(['START_OF_SENTENCE', 'END_OF_SENTENCE'])

    def combination_gen(self, sentence, comb=2):
        """Generate all possible combination in a sentence with the length of combination"""

        # With SOS
        output = []
        for i in range(-1 , len(sentence) + 2 - comb): # [len(sentence + 2) + 2 - comb + 1] iterations
            tup = []
            for j in range(comb):
                if i+j < 0:
                    tup.append('START_OF_SENTENCE')
                elif i+j < len(sentence):
                    tup.append(sentence[i+j])
                else: # There can't be two steps exceeds the length of the sentence
                    tup.append('END_OF_SENTENCE')
            output.append(tuple(tup))
        return output

        # # Without SOS
        # output = []
        # for i in range(len(sentence) + 2 - comb + 1): # [len(sentence + 2) + 2 - comb + 1] iterations
        #     tup = []
        #     for j in range(comb):
        #         if i+j < len(sentence):
        #             tup.append(sentence[i+j])
        #         else:
        #             tup.append('END_OF_SENTENCE')
        #     output.append(tuple(tup))
        # return output

    def denominator(self, model, word):
        """Calcualte the total number of the apperance of a denominator word in conditional prob"""
        total = 0.0
        for key in model[word].keys():
            total += model[word][key]
        return total

class Ngram_baseline(LangModel):
    def __init__(self, comb=2, unk_prob=0.0001):
        self.model = dict()
        self.unigram_model = dict()
        self.bigram_model=dict()
        self.lunk_prob = log(unk_prob, 2)
        self.very_small_value = 0.0001
        self.comb = comb

    def inc_word_mat_bi(self, token):
        """"""
        if token in self.bigram_model:
            self.bigram_model[token] += 1.0
        else:
            self.bigram_model[token] = 1.0

    def inc_word_mat(self, token):
        """Count the number of appearance of each word (macro word matrix)"""
        if token in self.model:
            self.model[token] += 1.0
        else:
            self.model[token] = 1.0

    def inc_word(self, w):
        """Count the number of appearance of each word (macro word matrix)"""
        if w in self.unigram_model:
            self.unigram_model[w] += 1.0
        else:
            self.unigram_model[w] = 1.0

    def fit_sentence(self, sentence):
        """Call inc_word() for a sentence and add 'EOS' to the macro word matrix
        Loop through all the possible combination of the bigram"""
        self.inc_word('START_OF_SENTENCE')
        self.inc_word_mat_bi(('START_OF_SENTENCE', 'START_OF_SENTENCE'))
        for token in self.combination_gen(sentence, comb=self.comb):
            self.inc_word_mat(token)
            # self.inc_word_mat_bi(token[:self.comb])
        for w in sentence:
            self.inc_word(w)
        for token in self.combination_gen(sentence, comb=2):
            self.inc_word_mat_bi(token)
        self.inc_word('END_OF_SENTENCE')
        self.inc_word_mat_bi(('END_OF_SENTENCE', 'END_OF_SENTENCE'))

    def norm(self):
        """Normalize and convert to log2-probs. (Strange definition of 'normalize')"""
        # pdb.set_trace()
        # for key in self.unigram_model.keys():
        #     if self.unigram_model[key] > 2:
        #         print(key, self.unigram_model[key])
        if self.comb == 2:
            for key in self.model.keys():
                denominator = self.unigram_model[key[0]]
                l_denom = log(denominator, 2)
                self.model[key] = log(self.model[key], 2) - l_denom
        elif self.comb == 3:
            for key in self.model.keys():
                # print(key[:self.comb-1])
                denominator = self.bigram_model[key[:self.comb-1]]
                l_denom = log(denominator, 2)
                self.model[key] = log(self.model[key], 2) - l_denom
        else:
            print("Unavailable ngram!")
            exit()

    def cond_logprob(self, word, previous, numOOV):
        # assert numOOV != 0, "numOOV == 0!"
        # if numOOV == 0:
        #     pdb.set_trace()
        # pdb.set_trace()

        # form the conditional / previous bigram (for trigram)
        if len(previous) < self.comb-1:
            # if previous == [] and word == 'START_OF_SENTENCE':
            #     pdb.set_trace()
            for _ in range(self.comb - 1 - len(previous)):
                previous = ['START_OF_SENTENCE'] + previous
            cond = previous
        elif len(previous) == self.comb-1:
            cond = previous
        else:
            cond = tuple([x for x in previous[-self.comb+1:len(previous)]])

        if tuple(list(cond) + [word]) in self.model:
            return self.model[tuple(list(cond) + [word])]
        else:
            try:
                return self.lunk_prob-log(numOOV, 2)
            except:
                # pdb.set_trace()
                return self.lunk_prob-log(1, 2)

    def vocab(self):
        # return self.unigram_model.keys()
        return set(self.unigram_model.keys()) - set(['START_OF_SENTENCE', 'END_OF_SENTENCE'])

    def combination_gen(self, sentence, comb=3):
        """Generate all possible combination in a sentence with the length of combination"""
        output = []
        for i in range(-comb+1, len(sentence)+comb-2, 1): # [len(sentence) + 2 - comb + 1] iterations
            tup = []
            for j in range(comb):
                if i+j < 0:
                    tup.append('START_OF_SENTENCE')
                elif i+j < len(sentence):
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

        self.unigram_model = dict()
        self.bigram_model=dict()
        self.temp = dict()
        self.lunk_prob = log(unk_prob, 2)
        self.lamb = 1.0 # for laplace smoothing
        self.comb = comb
        self.gamma = 2.0

        self.smoothing = False

    def inc_word_mat_bi(self, token):
        """"""
        if token in self.bigram_model:
            self.bigram_model[token] += 1.0
        else:
            self.bigram_model[token] = 1.0

    def inc_word_mat(self, token):
        """Count the number of appearance of each word (macro word matrix)"""
        if token in self.trigram_model:
            self.trigram_model[token] += 1.0
        else:
            self.trigram_model[token] = 1.0

    def inc_word(self, w):
        """Count the number of appearance of each word (macro word matrix)"""
        if w in self.unigram_model:
            self.unigram_model[w] += 1.0
        else:
            self.unigram_model[w] = 1.0

    def fit_sentence(self, sentence):
        """Call inc_word() for a sentence and add 'EOS' to the macro word matrix
        Loop through all the possible combination of the bigram"""
        self.inc_word('START_OF_SENTENCE')
        self.inc_word_mat_bi(('START_OF_SENTENCE', 'START_OF_SENTENCE'))
        for token in self.combination_gen(sentence, comb=self.comb):
            self.inc_word_mat(token)
            # self.inc_word_mat_bi(token[:self.comb])
        for w in sentence:
            self.inc_word(w)
        for token in self.combination_gen(sentence, comb=2):
            self.inc_word_mat_bi(token)
        self.inc_word('END_OF_SENTENCE')
        self.inc_word_mat_bi(('END_OF_SENTENCE', 'END_OF_SENTENCE'))

    def norm(self):
        """Normalize and convert to log2-probs. (Strange definition of 'normalize')"""
        # max = (None, 0)
        # for key in self.model.keys():
        #     if key != ('END_OF_SENTENCE', 'END_OF_SENTENCE'):
        #         if self.model[key] > max[1]:
        #             max = (key, self.model[key])
        # print(max)

        self.deal_unk()
        # pdb.set_trace()

        if self.comb == 2:
            for key in self.model.keys():
                # laplace smoothing
                if self.smoothing:
                    numerator = self.model[key] + self.lamb
                    denominator = self.unigram_model[key[0]] + self.lamb * len(self.vocab())
                else:
                    numerator = self.model[key]
                    denominator = self.unigram_model[key[0]]
                l_numer = log(numerator, 2)
                l_denom = log(denominator, 2)
                self.model[key] = l_numer - l_denom
        elif self.comb == 3:
            for key in self.model.keys():
                if self.smoothing:
                    numerator = self.model[key] + self.lamb
                    denominator = self.bigram_model[key[:self.comb-1]] + self.lamb * len(self.vocab())
                else:
                    numerator = self.model[key]
                    denominator = self.bigram_model[key[:self.comb-1]]
                l_numer = log(numerator, 2)
                l_denom = log(denominator, 2)
                self.model[key] = l_numer - l_denom
        else:
            print("Unavailable ngram!")
            exit()

    def cond_logprob(self, word, previous, numOOV):
        # assert numOOV != 0, "numOOV == 0!"
        # if numOOV == 0:
        #     pdb.set_trace()
        # pdb.set_trace()
        if len(previous) < self.comb-1:
            # if previous == [] and word == 'START_OF_SENTENCE':
            #     pdb.set_trace()
            for _ in range(self.comb - 1 - len(previous)):
                previous = ['START_OF_SENTENCE'] + previous
            cond = previous
        elif len(previous) == self.comb-1:
            cond = previous
        else:
            cond = tuple([x for x in previous[-self.comb+1:len(previous)]])
        if tuple(list(cond) + [word]) in self.model:
            return self.model[tuple(list(cond) + [word])]
        else:
            try:
                return self.lunk_prob-log(numOOV, 2)
            except:
                # pdb.set_trace()
                return self.lunk_prob-log(1, 2)

    def vocab(self):
        return self.unigram_model.keys()
        # return set(self.unigram_model.keys()) - set(['START_OF_SENTENCE', 'END_OF_SENTENCE'])

    def combination_gen(self, sentence, comb=3):
        """Generate all possible combination in a sentence with the length of combination"""
        output = []
        for i in range(-comb+1, len(sentence), 1): # [len(sentence) + 2 - comb + 1] iterations
            tup = []
            for j in range(comb):
                if i+j < 0:
                    tup.append('START_OF_SENTENCE')
                elif i+j < len(sentence):
                    tup.append(sentence[i+j])
                else:
                    tup.append('END_OF_SENTENCE')
            output.append(tuple(tup))
        return output

    def deal_unk(self):
        """"""
        if self.comb == 3:
            # rare_trigrams = []
            for token in self.trigram_model.keys():
                tup = []
                for i in range(self.comb):
                    if self.unigram_model[token[i]] < self.gamma:
                        tup.append('UNK')
                    else:
                        tup.append(token[i])
                if 'UNK' in tup:
                    # rare_trigrams.append(token)
                    if tuple(tup) in self.temp:
                        self.temp[tuple(tup)] = [self.temp.get(tuple(tup))[0] + [token], self.temp.get(tuple(tup))[1] + self.trigram_model.get(token)]
                    else:
                        self.temp[tuple(tup)] = [[token], self.trigram_model.get(token)]
            for token in self.temp.keys():
                for rare_tri in self.temp[token][0]:
                    self.trigram_model[rare_tri] = self.temp.get(token)[1]

            self.temp = dict()
            for token in self.bigram_model.keys():
                tup = []
                for i in range(self.comb - 1):
                    if self.unigram_model[token[i]] < self.gamma:
                        tup.append('UNK')
                    else:
                        tup.append(token[i])
                if 'UNK' in tup:
                    if tuple(tup) in self.temp:
                        self.temp[tuple(tup)] = [self.temp.get(tuple(tup))[0] + [token], self.temp.get(tuple(tup))[1] + self.bigram_model.get(token)]
                    else:
                        self.temp[tuple(tup)] = [[token], self.bigram_model.get(token)]
            for token in self.temp.keys():
                for rare_bi in self.temp[token][0]:
                    self.bigram_model[rare_bi] = self.temp.get(token)[1]

            self.model = self.trigram_model.copy()

        else: pass

class Ngram(LangModel):
    def __init__(self, comb=3, unk_prob=0.0001):
        self.model = dict()
        self.trigram_model = dict()
        self.unigram_model = dict()
        self.unk_model = dict()
        # self.bigram_model=dict()
        self.temp = dict()
        self.rare_words = list()
        self.lunk_prob = log(unk_prob, 2)
        self.lamb = 1.0 # for laplace smoothing
        self.comb = comb
        self.gamma = 10.0
        self.vocab_size = 0
        self.vocab_cache = None

        self.smoothing = True
        self.unk = True

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        if self.unk:
            corpus = [list(map(lambda x: 'UNK' if x in self.rare_words else x, s)) for s in corpus]
        vocab_set = set(self.vocab())
        words_set  = set([w for s in corpus for w in s])
        numOOV = len(words_set - vocab_set) # out-of-vocabulary
        
        return pow(2.0, self.entropy(corpus, numOOV))

    def replace(self, s):
        # return ['UNK' if w in self.rare_words else w for w in s]
        new_s = []
        for w in s:
            if w in self.rare_words:
                new_s.append('UNK')
            else:
                new_s.append(w)
        return new_s

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        
        self.build_unigram(corpus)
        self.find_rare_words()
        start_time = time.time()
        if self.unk:
            # new_corpus = []
            # count = 0
            start_time = time.time()
            corpus = [list(map(lambda x: 'UNK' if x in self.rare_words else x, s)) for s in corpus]

            # for s in corpus:
            #     count += 1
            #     new_s = ['UNK' if w in self.rare_words else w for w in s]
            #     new_corpus.append(new_s)
            #     print(count, '/', len(corpus), end='\r')
            # corpus = new_corpus
            print("Time for replace UNK in corpus:", time.time() - start_time, "sec")
        self.unigram_model = dict()
        self.build_unigram(corpus)
        self.vocab_cache = self.vocab()
        s_count = 0
        for s in corpus:
            if self.unk:
                self.deal_unk(s)
                self.temp = dict(self.unk_model)
            else:
                self.fit_sentence(s)
                self.temp = dict(self.model)
            s_count += 1
        
        self.norm()

    def inc_word_mat(self, token):
        """Count the number of appearance of each word (macro word matrix)"""
        previous = (token[-2], token[-1])
        if previous in self.model:
            if token[-1] in self.model[previous]:
                self.model[previous][token[-1]] += 1.0
            else:
                self.model[previous][token[-1]] = 1.0
        else:
            self.model[previous] = {token[-1]: 1.0}

    def inc_word(self, w):
        """Count the number of appearance of each word (macro word matrix)"""
        if w in self.unigram_model:
            self.unigram_model[w] += 1.0
        else:
            self.unigram_model[w] = 1.0

    def build_unigram(self, corpus):
        for s in corpus:
            self.inc_word('START_OF_SENTENCE')
            for w in s:
                self.inc_word(w)
            self.inc_word('END_OF_SENTENCE')
        self.vocab_size = len(self.vocab())
        self.vocab_cache = set(self.vocab())

    def fit_sentence(self, sentence):
        """"""
        for token in self.combination_gen(sentence, comb=self.comb):
            self.inc_word_mat(token)
        
    def norm(self):
        """Normalize and convert to log2-probs. (Strange definition of 'normalize')"""
        # max = (None, 0)
        # for key in self.model.keys():
        #     if key != ('END_OF_SENTENCE', 'END_OF_SENTENCE'):
        #         if self.model[key] > max[1]:
        #             max = (key, self.model[key])
        # print(max)

        # self.deal_unk()

        if self.comb == 2:
            for key in self.model.keys():
                # laplace smoothing
                if self.smoothing:
                    numerator = self.model[key] + self.lamb
                    denominator = self.unigram_model[key[0]] + self.lamb * self.vocab_size + 2 # SOS and EOS
                else:
                    numerator = self.model[key]
                    denominator = self.unigram_model[key[0]]
                l_numer = log(numerator, 2)
                l_denom = log(denominator, 2)
                self.model[key] = l_numer - l_denom

        # The only part should work for now
        elif self.comb == 3:
            if self.unk:
                pdb.set_trace()
                for bigram in self.unk_model.keys():
                    denominator = self.denominator_gen(self.unk_model[bigram])
                    for w in self.unk_model[bigram].keys():
                        if self.smoothing:
                            numerator = self.unk_model[bigram][w] + self.lamb
                            denominator =  denominator + self.lamb * self.vocab_size
                        else:
                            numerator = self.unk_model[bigram][w]
                            denominator = denominator
                        l_numer = log(numerator, 2)
                        l_denom = log(denominator, 2)
                        self.unk_model[bigram][w] = l_numer - l_denom
                    self.unk_model[bigram]['AMZN_ALEXA'] = - l_denom

            else:
                for bigram in self.model.keys():
                    for w in self.model[bigram].keys():
                        if self.smoothing:
                            numerator = self.model[bigram][w] + self.lamb
                            denominator = self.denominator_gen(self.temp[bigram]) + self.lamb * self.vocab_size
                        else:
                            numerator = self.model[bigram][w]
                            denominator = self.denominator_gen(self.temp[bigram])
                        l_numer = log(numerator, 2)
                        l_denom = log(denominator, 2)
                        self.model[bigram][w] = l_numer - l_denom
                    self.model[bigram]['AMZN_ALEXA'] = - l_denom

        else:
            print("Unavailable ngram!")
            exit()

    def cond_logprob(self, word, previous, numOOV):
        # assert numOOV != 0, "numOOV == 0!"
        # if numOOV == 0:
        #     pdb.set_trace()
        pdb.set_trace()
        if self.unk:
            # check word and previous if needing unk
            if False:
                return 0
            if word in self.vocab_cache:
                pass
            else:
                word = 'UNK'

            if len(previous) < self.comb-1:
                for _ in range(self.comb - 1 - len(previous)):
                    previous = ['START_OF_SENTENCE'] + previous
                cond = tuple(previous)
            elif len(previous) == self.comb-1:
                cond = tuple(previous)
            else:
                cond = tuple([x for x in previous[-self.comb+1:len(previous)]])

            if cond in self.unk_model.keys():
                if word in self.vocab_cache:
                    if word in self.unk_model[cond].keys():
                        return self.unk_model[cond][word]
                    else:
                        return self.unk_model[cond]['AMZN_ALEXA']
                else:
                    if 'UNK' in self.unk_model[cond]:
                        return self.unk_model[cond]['UNK'] - log(len(self.rare_words), 2)
                    else:
                        return self.unk_odel[cond]['AMZN_ALEXA']
            else:
                if word in self.vocab_cache:
                    if word in self.unk_model[('UNK', 'UNK')].keys():
                        return self.unk_model[('UNK', 'UNK')][word]
                    else:
                        return self.unk_model[('UNK', 'UNK')]['AMZN_ALEXA']
                else:
                    if 'UNK' in self.unk_model[('UNK', 'UNK')]:
                        return self.unk_model[('UNK', 'UNK')] - log(len(self.rare_words), 2)
                    else:
                        return self.unk_model[('UNK', 'UNK')]['AMZN_ALEXA']
                # try:
                #     return self.lunk_prob-log(numOOV, 2)
                # except:
                #     # pdb.set_trace()
                #     return self.lunk_prob-log(1, 2)
        else:
            if len(previous) < self.comb-1:
                for _ in range(self.comb - 1 - len(previous)):
                    previous = ['START_OF_SENTENCE'] + previous
                cond = tuple(previous)
            elif len(previous) == self.comb-1:
                cond = tuple(previous)
            else:
                cond = tuple([x for x in previous[-self.comb+1:len(previous)]])

            if cond in self.model:
                if word in self.model[cond]:
                    return self.model[cond][word]
                else:
                    return self.model[cond]['AMZN_ALEXA']

            else:
                try:
                    return self.lunk_prob-log(numOOV, 2)
                except:
                    # pdb.set_trace()
                    return self.lunk_prob-log(1, 2)

    def vocab(self):
        return self.unigram_model.keys()
        # return set(self.unigram_model.keys()) - set(['START_OF_SENTENCE', 'END_OF_SENTENCE'])

    def combination_gen(self, sentence, comb=3):
        """Generate all possible combination in a sentence with the length of combination"""
        output = []
        for i in range(-comb+1, len(sentence), 1): # [len(sentence) + 2 - comb + 1] iterations
            tup = []
            for j in range(comb):
                if i+j < 0:
                    tup.append('START_OF_SENTENCE')
                elif i+j < len(sentence):
                    tup.append(sentence[i+j])
                else:
                    tup.append('END_OF_SENTENCE')
            output.append(tuple(tup))
        return output

    def denominator_gen(self, dic):
        """Calcualte the sum of the number of words given a bigram as previous"""
        total = 0.0
        for key in dic.keys():
            total += dic[key]
        if total < 0:
            pdb.set_trace()
        return total

    def find_rare_words(self):
        for word in self.unigram_model:
            if self.unigram_model[word] < self.gamma:
                self.rare_words.append(word)
        self.rare_words = set(self.rare_words)

    def replace_with_UNK(self, sentence):
        return ['UNK' if x in self.rare_words else x for x in sentence]
        # return output

    def deal_unk(self, sentence):
        """"""
        # sentence = self.replace_with_UNK(sentence)
        if self.comb == 3:
            for token in self.combination_gen(sentence):
                # increment the unk_model
                previous = (token[-3], token[-2])
                if previous in self.unk_model:
                    if token[-1] in self.unk_model[previous]:
                        self.unk_model[previous][token[-1]] += 1.0
                    else:
                        self.unk_model[previous][token[-1]] = 1.0
                else:
                    self.unk_model[previous] = {token[-1]: 1.0}

        else: pass

