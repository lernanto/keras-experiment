# -*- encoding: utf-8 -*-

"""
Utilities to deal with text.
"""

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import logging
import codecs
import collections


logger = logging.getLogger('util')


class Corpus:
    """
    Get corpus from file.
    """

    def __init__(self, fname, encoding='utf-8'):
        self.file = fname
        self.encoding = 'utf-8'

    def __iter__(self):
        """
        Get tokens from corpus.
        """

        with codecs.open(self.file, encoding=self.encoding) as f:
            for line in f:
                yield line.rstrip()


class Dictionary:
    """
    Dictionary to convert characters to ID's and vise versa.
    """

    PADDING = 0
    BOS = 1
    EOS = 2
    UNKNOWN = 3

    RESERVED_WORDS = ('', '<s>', '</s>', '??')

    def build(self, corpus, min_freq=10, max_size=10000):
        """
        Build from corpus.
        """

        count = collections.defaultdict(int)

        for sentence in corpus:
            for word in sentence:
                count[word] += 1

        self.word_to_id_dic = {}
        self.id_to_word_dic = list(self.RESERVED_WORDS)

        for word, freq in sorted(count.iteritems(), key=lambda x: x[1], reverse=True)[:max_size]:
            if freq >= min_freq:
                self.word_to_id_dic[word] = len(self.word_to_id_dic) + len(self.RESERVED_WORDS)
                self.id_to_word_dic.append(word)

        logger.debug('dictionary size = %d' % self.size())

    def load(self, fname):
        """
        Load dictionary from file.
        """

        self.word_to_id_dic = {}
        self.id_to_word_dic = list(self.RESERVED_WORDS)

        with open(fname) as f:
            for i, line in enumerate(f):
                word = line.strip()
                self.word_to_id_dic[word] = i + len(self.RESERVED_WORDS)
                self.id_to_word_dic.append(word)

        logger.debug('dictionary size = %d' % self.size())

    def save(self, fname):
        """
        Save dictionary to file.
        """

        with codecs.open(fname, 'w', encoding='utf-8') as f:
            for word in self.id_to_word_dic[len(self.RESERVED_WORDS):]:
                print >> f, word

    def size(self):
        """
        Get number of words, including achors.
        """

        return len(self.id_to_word_dic)

    def __len__(self):
        return self.size()

    def word_to_id(self, word):
        """
        Convert word to ID.
        """

        return self.word_to_id_dic.get(word, self.UNKNOWN)

    def id_to_word(self, id):
        """
        Convert ID to word.
        """

        return self.id_to_word_dic[id]
