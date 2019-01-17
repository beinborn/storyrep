import numpy as np
from language_models.abstract_text_encoder import TextEncoder


# This class generates truely random embeddings.
# Every word is assigned a random  vector of the requested dimensionality.
# THe difference to the random encoder is:
# The vector is changed every time, that means that the same word gets assigned different vectors in different occurrences.


class TruelyRandomEncoder(TextEncoder):
    def __init__(self, embedding_dir="", dimensions=512):
        super(TruelyRandomEncoder, self).__init__(embedding_dir)
        self.dimensions = dimensions
        # Set the seed to make experiments reproducible.
        self.seed = 42
        np.random.seed(self.seed)

    def get_word_embeddings(self, words, name="test"):
        embeddings = []
        for word in words:
            embedding = np.random.rand(self.dimensions)
            embeddings.append(embedding)

        return embeddings
