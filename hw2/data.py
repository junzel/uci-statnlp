#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import sys
import pdb


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


def textToTokens(text):
    """Converts input string to a corpus of tokenized sentences.

    Assumes that the sentences are divided by newlines (but will ignore empty sentences).
    You can use this to try out your own datasets, but is not needed for reading the homework data.
    """
    corpus = []
    sents = text.split("\n")
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(sents)
    tokenizer = count_vect.build_tokenizer()
    for s in sents:
        toks = tokenizer(s)
        if len(toks) > 0:
            corpus.append(toks)
    return corpus

def file_splitter(filename, seed = 0, train_prop = 0.7, dev_prop = 0.15,
    test_prop = 0.15):
    """Splits the lines of a file into 3 output files."""
    import random
    rnd = random.Random(seed)
    basename = filename[:-4]
    train_file = open(basename + ".train.txt", "w")
    test_file = open(basename + ".test.txt", "w")
    dev_file = open(basename + ".dev.txt", "w")
    with open(filename, 'r') as f:
        for l in f.readlines():
            p = rnd.random()
            if p < train_prop:
                train_file.write(l)
            elif p < train_prop + dev_prop:
                dev_file.write(l)
            else:
                test_file.write(l)
    train_file.close()
    test_file.close()
    dev_file.close()

def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    train_mem = tar.getmember(dname + ".train.txt")
    train_txt = unicode(tar.extractfile(train_mem).read(), errors='replace')
    test_mem = tar.getmember(dname + ".test.txt")
    test_txt = unicode(tar.extractfile(test_mem).read(), errors='replace')
    dev_mem = tar.getmember(dname + ".dev.txt")
    dev_txt = unicode(tar.extractfile(dev_mem).read(), errors='replace')

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(ngram_range=(2, 2))
    count_vect.fit(train_txt.split("\n")) # each sentence in the corpus is devided by '\n'
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
    print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
    return data

def learn_unigram(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Unigram
    unigram = Unigram()
    unigram.fit_corpus(data.train)
    print("vocab:", len(unigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", unigram.perplexity(data.train))
    print("dev  :", unigram.perplexity(data.dev))
    print("test :", unigram.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(unigram)
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['The'])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['The'])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['The'])))
    return unigram

def learn_bigram(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Bigram
    bigram = Bigram()
    bigram.fit_corpus(data.train)
    print("vocab:", len(bigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", bigram.perplexity(data.train))
    print("dev  :", bigram.perplexity(data.dev))
    print("test :", bigram.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(bigram)
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['START_OF_SENTENCE'])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['START_OF_SENTENCE'])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['START_OF_SENTENCE'])))
    return bigram

def learn_ngram(data, hyperp_set={}):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Ngram
    # from lm import Ngram_baseline
    if hyperp_set == {}:
        ngram = Ngram(comb=3)
    else:
        if list(hyperp_set.keys())[0] == 'lamb':
            print(hyperp_set[list(hyperp_set.keys())[0]])
            ngram = Ngram(comb=3, lamb=hyperp_set[list(hyperp_set.keys())[0]])
        elif list(hyperp_set.keys())[0] == 'gamma':
            ngram = Ngram(comb=3, gamma=hyperp_set[list(hyperp_set.keys())[0]])
        else:
            ngram = Ngram(comb=3)
        
    ngram.fit_corpus(data.train)
    print("vocab:", len(ngram.vocab()))
    # evaluate on train, test, and dev
    print("train:", ngram.perplexity(data.train))
    print("dev  :", ngram.perplexity(data.dev))
    print("test :", ngram.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(ngram)
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['The'])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['The'])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['The'])))
    return ngram

def print_table(table, row_names, col_names, latex_file = None):
    """Pretty prints the table given the table, and row and col names.

    If a latex_file is provided (and tabulate is installed), it also writes a
    file containing the LaTeX source of the table (which you can \\input into your report)
    """
    try:
        from tabulate import tabulate
        rows = [*map(lambda rt: [rt[0]] + rt[1], zip(row_names,table.tolist()))]
        print(tabulate(rows, headers = [""] + col_names))
        if latex_file is not None:
            latex_str = tabulate(rows, headers = [""] + col_names, tablefmt="latex")
            with open(latex_file, 'w') as f:
                f.write(latex_str)
                f.close()
    except ImportError as e:
        row_format ="{:>15} " * (len(col_names) + 1)
        print(row_format.format("", *col_names))
        for row_name, row in zip(row_names, table):
            print(row_format.format(row_name, *row))

if __name__ == "__main__":
    # Do no run, the following function was used to generate the splits
    # file_splitter("data/reuters.txt")

    # hyperp_set = {'lamb': [0.1, 0.5, 1.0, 2.0, 5.0]}
    # hyperp_set = {'gamma': [2, 5, 10, 15, 20]}
    hyperp_set = {}

    if hyperp_set == {}:
        dnames = ["brown", "reuters", "gutenberg"]
        datas = []
        models = []
        word_count = {}
        # Learn the models for each of the domains, and evaluate it
        for dname in dnames:
            print("-----------------------")
            print(dname)
            data = read_texts("data/corpora.tar.gz", dname)
            datas.append(data)
            # model = learn_ngram(data, hyperp_set)
            model = learn_unigram(data)
            word_count[dname] = model.printout_model
            models.append(model)
        if True:
            import pickle
            with open('word_count.pkl', 'wb') as f:
                pickle.dump(word_count, f)
        
        # compute the perplexity of all pairs
        n = len(dnames)
        perp_dev = np.zeros((n,n))
        perp_test = np.zeros((n,n))
        perp_train = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                perp_dev[i][j] = models[i].perplexity(datas[j].dev)
                perp_test[i][j] = models[i].perplexity(datas[j].test)
                perp_train[i][j] = models[i].perplexity(datas[j].train)

        print("-------------------------------")
        print("x train")
        print_table(perp_train, dnames, dnames, "table-train.tex")
        print("-------------------------------")
        print("x dev")
        print_table(perp_dev, dnames, dnames, "table-dev.tex")
        print("-------------------------------")
        print("x test")
        print_table(perp_test, dnames, dnames, "table-test.tex")
    else:
        results = {}
        for hp in hyperp_set[list(hyperp_set.keys())[0]]:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('                      '+list(hyperp_set.keys())[0]+str(hp)+'                       ')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            dnames = ["brown", "reuters", "gutenberg"]
            datas = []
            models = []
            # Learn the models for each of the domains, and evaluate it
            for dname in dnames:
                print("-----------------------")
                print(dname)
                data = read_texts("data/corpora.tar.gz", dname)
                datas.append(data)
                model = learn_ngram(data, {list(hyperp_set.keys())[0]: hp})
                models.append(model)
            # compute the perplexity of all pairs
            n = len(dnames)
            perp_dev = np.zeros((n,n))
            perp_test = np.zeros((n,n))
            perp_train = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    perp_dev[i][j] = models[i].perplexity(datas[j].dev)
                    perp_test[i][j] = models[i].perplexity(datas[j].test)
                    perp_train[i][j] = models[i].perplexity(datas[j].train)
            results[list(hyperp_set.keys())[0]+str(hp)] = {'dev': perp_dev, 'test': perp_test, 'train': perp_train}

            print("-------------------------------")
            print("x train")
            print_table(perp_train, dnames, dnames, list(hyperp_set.keys())[0]+str(hp)+"table-train.tex")
            print("-------------------------------")
            print("x dev")
            print_table(perp_dev, dnames, dnames, list(hyperp_set.keys())[0]+str(hp)+"table-dev.tex")
            print("-------------------------------")
            print("x test")
            print_table(perp_test, dnames, dnames, list(hyperp_set.keys())[0]+str(hp)+"table-test.tex")

        import pickle
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)
