import numpy as np


from language_models.abstract_text_encoder import TextEncoder

# This class generates random embeddings.
# Every word is assigned a random (but fixed) vector of the requested dimensionality.


class RandomEncoder(TextEncoder):
    def __init__(self, embedding_dir, dimensions=512):
        super(RandomEncoder, self).__init__(embedding_dir)
        self.dimensions = dimensions
        self.dict = {}

        # Set the seed to make experiments reproducible.
        self.seed = 5
        np.random.seed(self.seed)

    def get_word_embeddings(self,  words, name="test"):
        embeddings = []
        for word in words:
            if word in self.dict.keys():
                embeddings.append(self.dict[word])
            else:
                embedding = np.random.rand(self.dimensions)
                self.dict[word] = embedding
                embeddings.append(embedding)


        return embeddings
