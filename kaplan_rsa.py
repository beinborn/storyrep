from read_dataset.read_stories_data import StoryDataReader
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
save_dir = user_dir + "/Experiments/multilingual/"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the language models
    dir = "/Users/lisa/Experiments/multilingual/"
    bertencoder = BertEncoder(save_dir + "Bert/")
    elmoencoder = ElmoEncoder(save_dir + "Elmo/")
    random_wordbased_encoder = RandomEncoder(save_dir + "RandomWordBased/", dimensions=1024)
    truely_random_encoder = TruelyRandomEncoder("", dimensions=1024)
    # for language in ["english", "farsi", "chinese"]:
    kaplan_reader = StoryDataReader(data_dir=kaplan_dir, language="english")
    subjects = [x for x in range(0, 28)]
    story_data = kaplan_reader.read_all_events(subjects)

    subject_scans = {}
    labels = []
    # Collect scans
    for subject in story_data.keys():
        scans = []
        for block in story_data[subject]:
            scans.append(block.scan_events[0].scan)
        subject_scans[subject] = scans
        labels.append("Subject_" + str(subject))
        # I tested clustering the stories based on the scans with and without applying PCA.
        # Resulting clusters are very diverse, you can ignore them for now.

        # print("Run clustering")
        # kmeans = KMeans(n_clusters=5, random_state=0).fit(scans)
        # y_kmeans = kmeans.predict(scans)
        # print(y_kmeans)
        # print("Plot reduced scans")
        # plt.scatter(reduced_scans[:, 0], reduced_scans[:, 1], c=y_kmeans, s=50, cmap='viridis')
        # # Add story indices as labels
        # for i in range(0, len(reduced_scans)):
        #     x = reduced_scans[i, 0]
        #     y = reduced_scans[i, 1]
        #     plt.text(x+0.1, y+0.1, i, fontsize=9)
        # plt.title(str(subject))
        # plt.show()
        # #Print clusters
        # for label in range(0,5):
        #     print("Stories in Cluster " + str(label))
        #     print(np.where(kmeans.labels_ == label))
        # print(subject)
        # print()

    # Get embeddings
    # Stimuli are the same for all subjects, so we just take them from the last one
    stimuli = []
    for block in story_data[subject]:
        stimuli.append(block.sentences)

    embeddings_container = []

    # We are playing around with this."
    sentence_pooling = "sum"
    story_pooling = "sum"
    for encoder in (bertencoder, elmoencoder, random_wordbased_encoder, truely_random_encoder):
        embeddings = encoder.get_story_embeddings(stimuli, name="Stories_english/concat",
                                                  sentence_pooling_mode=sentence_pooling,
                                                  story_pooling_mode=story_pooling)
        embeddings_container.append(embeddings)
        labels.append(encoder.__class__.__name__)

    data = [subject_scans[x] for x in subjects]
    data.extend([x for x in embeddings_container])
    reduced_data = []
    print(embeddings_container.count(embeddings_container[0]) == len(embeddings_container))
    # Test: Apply pca on scans to reduce the dimensionality
    for rep in data:
        print("Run PCA")
        pca = PCA(n_components=40)
        print("Fit PCA")
        pca.fit(rep)  # find the principal components
        print("Transform scans")
        reduced_data.append(pca.transform(rep))

    # These two functions automatically produce plots.
    # Edit evaluation.metrics to suppress them or add more detailed plots.
    x, C = get_dists(data, labels)
    spearman, pearson, kullback = compute_distance_over_dists(x, C, labels)