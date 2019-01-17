import numpy as np


# Apply a pooling functions on a list of k embeddings of dimensionality n to return a single embedding
def pool_embeddings(embeddings, pooling_mode):
    # Mean over all embeddings, dimensionality = n
    if pooling_mode == "mean":
        return np.mean(embeddings, axis=0)

    # Return sum over all embeddings, dimensionality = n
    elif pooling_mode == "sum":
        return np.sum(embeddings, axis=0)

    # Return last embedding, dimensionality = n
    # In recurrent networks, it could be assumed that the last embedding contains all the important information from the previous ones
    elif pooling_mode == "final":
        return embeddings[-1]


    # Careful! This returns dimensionality k * n,
    # which means the dimensionality differs depending on k

    elif pooling_mode == "concat":
        print(np.asarray(embeddings).shape)
        print(np.concatenate(embeddings, axis=None).shape)
        # TO make uniform, we only concat the last 5 sentences of the story
        return np.concatenate(embeddings[-5], axis=None)
