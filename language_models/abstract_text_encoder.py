# TODO: adjust this if necessary

import os
import pickle
import logging
from language_models.utils import pool_embeddings


class TextEncoder(object):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir
        # Set this to False if you want to do a lot of experimenting and not delete the produced embeddings every time
        self.load_previous = True

    def get_word_embeddings(self, text):
        raise NotImplementedError()

    # Here, a sentence embedding is just a list of word embeddings.
    # Contextualized encoders should override this method!
    def get_sentence_embeddings(self, sentences, name="test"):
        sentence_embeddings = []
        for sentence in sentences:
            word_embeddings = self.get_word_embeddings(sentence, name="test")
            sentence_embeddings.append(word_embeddings)
        return sentence_embeddings

    def get_story_embeddings(self, stories, name="test", sentence_pooling_mode="mean", story_pooling_mode="mean"):
        embedding_file = self.embedding_dir + name + "_story_embeddings.pickle"
        story_embeddings = []

        # Careful, if the file exists, I load it. Make sure to delete it, if you want to reencode.
        # if self.load_previous and os.path.isfile(embedding_file):
        #
        #     story_embeddings = self.load_embeddings(embedding_file)

        i = 0
        for story in stories:
            sentence_embeddings = self.get_sentence_embeddings(story, name=name + "_" + str(i))
            pooled_sentence_embeddings = []
            for embedding in sentence_embeddings:
                pooled_sentence_embeddings.append(pool_embeddings(embedding, sentence_pooling_mode))

            story_embeddings.append(pool_embeddings(pooled_sentence_embeddings, story_pooling_mode))
            i += 1

        self.save_embeddings(embedding_file, story_embeddings)

        return story_embeddings


    def load_embeddings(self, embedding_file):
        if self.load_previous and os.path.isfile(embedding_file):
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                embeddings = pickle.load(handle)
            return embeddings
        else:
            return []


    def save_embeddings(self, embedding_file, embeddings):
        os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
        with open(embedding_file, 'wb') as handle:
            pickle.dump(embeddings, handle)
