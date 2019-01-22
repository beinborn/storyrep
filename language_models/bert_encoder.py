import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from language_models.abstract_text_encoder import TextEncoder


# This class retrieves embeddings from the multilingual BertEncoder
class BertEncoder(TextEncoder):
    def __init__(self, embedding_dir, model_name="bert-base-multilingual-cased", layer=-2):
        super(BertEncoder, self).__init__(embedding_dir)

        # Load pre-trained model (weights) and set to evaluation mode (no more training)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

        # Load word piece tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Layer from which to get the embeddings
        self.layer = layer

    def get_word_embeddings(self,  words, name ="test"):
        raise NotImplementedError(
            "It does not really make sense to get word embeddings without context from Bert because we do not know which language they would come from.")

    def get_sentence_embeddings(self, sentences, name ="test", layer=-2):
        # Load any preexisting embeddings for name. Careful, make sure that name is unique!
        embedding_file = self.embedding_dir + name + "sentence_embeddings.pickle"
        embeddings = self.load_embeddings(embedding_file)

        if not (len(embeddings) == len(sentences)):
            embeddings = []
            for sentence in sentences:
                # Bert uses its own "word piece tokenization"
                # It does not make sense to tokenize in the reader, then detokenize here and then tokenize again.
                # If I go on with that, I should probably not do tokenization in the reader.
                untokenized = " ".join(sentence)
                tokenized = self.tokenizer.tokenize(untokenized)
                # Convert token to vocabulary indices
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)

                # Convert inputs to PyTorch tensors
                segment_ids = [0 for token in tokenized]
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensor = torch.tensor([segment_ids])

                # Predict hidden states features for each layer
                encoded_layers, _ = self.model(tokens_tensor, segments_tensor)

                assert len(encoded_layers) == 12

                # Which layer and which pooling function should we use for fixed sentence reperesentations?
                # 1. Jacob Devlin on: https://github.com/google-research/bert/issues/71
                # "If you want sentence representation that you don't want to train,
                # your best bet would just to be to average all the final hidden layers of all of the tokens in the sentence
                #  (or second-to-last hidden layers, i.e., -2, would be better)."
                # 2. In the paper, they say that concatenating the top four layers for each token could also be a good representation.
                # 3. In Bert as a service, they use the second to last layer and do mean pooling
                embeddings.append(encoded_layers[layer][0].detach().numpy())
        self.save_embeddings(embedding_file, embeddings)
        return embeddings

xs
