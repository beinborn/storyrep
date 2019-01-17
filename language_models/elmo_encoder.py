from allennlp.commands.elmo import ElmoEmbedder
import logging
from language_models.abstract_text_encoder import TextEncoder


# This class retrieves embeddings from the Elmo model by Peters et al (2018).
# See https://allennlp.org/elmo for details.
# The three methods for getting embeddings for sentences, words, stories are slightly repetitive.
# They could be combined into a single method, but I found it easier to keep them separate for debugging.
# Note: If pickle files exist in the embedding dir, the encoder automatically reads them.
# Make sure to delete them, if you want new embeddings.


class ElmoEncoder(TextEncoder):

    def __init__(self, embedding_dir):
        super(ElmoEncoder, self).__init__(embedding_dir)
        self.layer_id = 1
        self.only_forward = False
        self.embedder = None

    # Word embeddings are just sentence embeddings with sentences consisting of a single word
    # This is most likely too naive.
    def get_word_embeddings(self, words, name="test"):
        self.get_sentence_embeddings(name, words)

    # Takes a list of sentences and returns a list of embeddings
    def get_sentence_embeddings(self, sentences, name="test"):
        # Layer 0 are token representations which are not sensitive to context
        # Layer 1 are representations from the first bilstm
        # Layer 2 are the representations from the second bilstm

        # Load any preexisting embeddings for name. Careful, make sure that name is unique!
        embedding_file = self.embedding_dir + name + "sentence_embeddings.pickle"
        sentence_embeddings = self.load_embeddings(embedding_file)

        if not (len(sentence_embeddings) == len(sentences)):
            if self.embedder is None:
                self.embedder = ElmoEmbedder()

            sentence_embeddings = self.embedder.embed_batch(sentences)

            if not len(sentence_embeddings) == len(sentences):
                logging.info("Something went wrong with the embedding. Number of embeddings: " + str(
                    len(sentence_embeddings)) + " Number of sentences: " + str(len(sentences)))

            self.save_embeddings(embedding_file, sentence_embeddings)

        single_layer_embeddings = [embedding[self.layer_id] for embedding in sentence_embeddings[:]]

        if self.only_forward:
            forward_embeddings = []
            for sentence_embedding in single_layer_embeddings:
                forward_embeddings.append([token_embedding[0:512] for token_embedding in sentence_embedding])
            return forward_embeddings
        else:
            return single_layer_embeddings
