from read_dataset.read_stories_data import StoryDataReader
from read_dataset.read_pereira_data import PereiraReader
from language_models.random_encoder import RandomEncoder
from language_models.bert_encoder import BertEncoder
from language_models.elmo_encoder import ElmoEncoder
from language_models.truely_random_encoder import TruelyRandomEncoder
from mapping_models.ridge_regression_mapper import RegressionMapper

from evaluation.metrics import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Here, I am analyzing story representations with Bert
user_dir = "/Users/lisa/"

kaplan_dir = user_dir + "Corpora/Kaplan_data/"
pereira_dir =user_dir + "Corpora/pereira_data/"
save_dir = user_dir + "/Experiments/pereira/"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the language models
    # TODO: Play around with layer
    bertencoder = BertEncoder(save_dir + "Bert/")
    elmoencoder = ElmoEncoder(save_dir + "Elmo/")
    random_wordbased_encoder = RandomEncoder(save_dir + "RandomWordBased/", dimensions=1024)
    truely_random_encoder = TruelyRandomEncoder("", dimensions=1024)

    pereira_reader = PereiraReader(data_dir=pereira_dir, experiment=2)
    subjects = [2,4]
    story_data = pereira_reader.read_all_events(subjects)

    labels = []

    scans =[]
    stimuli = []
    for s in subjects:
        stim = []
        subject_scans =[]
        for block in story_data[s][0:20]:
            for event in block.scan_events:

                subject_scans.append(event.scan )
                stimulus =[block.sentences[sentence_id][token_id] for sentence_id, token_id in event.stimulus_pointers ]
                stim.append(stimulus)
        scans.append(subject_scans)
        labels.append("Subject_" + str(s))
# Are stimuli always the same for all participants?
    if not stim in stimuli:
        print("New set of stimuli: ")
        print(stim, stimuli)
        stimuli.append(stim)

    stimuli = stimuli[0]

    print(len(stimuli), len(scans))
    embeddings_container = []

    # We are playing around with this."
    sentence_pooling = "sum"
    story_pooling = "sum"
    for encoder in (bertencoder, elmoencoder, random_wordbased_encoder, truely_random_encoder):

        embeddings = encoder.get_story_embeddings(stimuli, name="Pererira/sum",
                                                  sentence_pooling_mode=sentence_pooling,
                                                  story_pooling_mode=story_pooling)
        embeddings_container.append(embeddings)
        labels.append(encoder.__class__.__name__)

    data = [x for x in scans]
    data.extend([x for x in embeddings_container])
    reduced_data = []
    print(embeddings_container.count(embeddings_container[0]) == len(embeddings_container))
    # Test: Apply pca on scans to reduce the dimensionality
    i = 0
    for rep in data:
        print(labels[i])
        print(np.asarray(rep).shape)
        i+=1
        print("Run PCA")
        pca = PCA(n_components=30)
        print("Fit PCA")
        pca.fit(rep)  # find the principal components
        print("Transform scans")
        reduced_data.append(pca.transform(rep))

    # These two functions automatically produce plots.
    # Edit evaluation.metrics to suppress them or add more detailed plots.
    x, C = get_dists(data, labels)
    spearman, pearson, kullback = compute_distance_over_dists(x, C, labels)
