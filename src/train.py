#!/usr/bin/env python
#-*- encoding: utf8 -*-

import rospkg
import rospy

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word

class BuildData():

    def __init__(self):
        rospy.loginfo('\033[94m[build_data]\033[0m initialized.')
        self.build()

    def build(self):
        # get config and processing of words
        config = Config(load=False)
        processing_word = get_processing_word(lowercase=True)

        # generators
        dev = CoNLLDataset(config.filename_dev, processing_word)
        test = CoNLLDataset(config.filename_test, processing_word)
        train = CoNLLDataset(config.filename_train, processing_word)

        # build word and tag vocab
        vocab_words, vocab_tags = get_vocabs([train, dev, test])
        vocab_glove = get_glove_vocab(config.filename_glove)

        vocab = vocab_words & vocab_glove
        vocab.add(UNK)
        vocab.add(NUM)

        # save vocab
        write_vocab(vocab, config.filename_words)
        write_vocab(vocab_tags, config.filename_tags)

        # trim GloVe Vectors
        vocab = load_vocab(config.filename_words)
        export_trimmed_glove_vectors(vocab, config.filename_glove,
                                    config.filename_trimmed, config.dim_word)

        # build and save char vocab
        train = CoNLLDataset(config.filename_train)
        vocab_chars = get_char_vocab(train)
        write_vocab(vocab_chars, config.filename_chars)

class Trainer():

    def __init__(self):
        b = BuildData()
        
        rospy.loginfo('\033[94m[%s]\033[0m initialized.'%rospy.get_name())
        self.train()

    def train(self):
        config = Config()

        # build model
        model = NERModel(config)
        model.build()

        # create dataset
        dev = CoNLLDataset(config.filename_dev, config.processing_word,
                            config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word,
                            config.processing_tag, config.max_iter)
        test = CoNLLDataset(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)

        model.train(train, dev)

if __name__ == '__main__':
    rospy.init_node('entity_tagging', anonymous=False)
    t = Trainer()