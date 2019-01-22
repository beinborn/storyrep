from read_dataset import read_alice_data
from read_dataset.read_harry_potter_data import HarryPotterReader
from read_dataset.read_stories_data import StoryDataReader
from read_dataset.read_alice_data import AliceDataReader
from read_dataset.read_words_data import WordsReader
from read_dataset.read_NBD_data import NBDReader
from  read_dataset.read_pereira_data import PereiraReader
from voxel_preprocessing.select_voxels import select_most_varied_voxels
from voxel_preprocessing.preprocess_voxels import reduce_mean
import numpy as np
from nilearn import plotting

# SET THE DATA DIR: 
data_dir = "/Users/lisa/Corpora/"


# ---- Mitchell DATA -----
# Make sure to get the data at http://www.cs.cmu.edu/~fmri/science2008/data.html
# Adjust the dir!

# print("\n\nMitchell Data")
# mitchell_reader = WordsReader(data_dir=data_dir +"mitchell/")
# subject_id = 1
# mitchell_data = mitchell_reader.read_all_events(subject_ids=[subject_id])
# all_scans = []
# for block in mitchell_data[subject_id][0:5]:
#     sentences = block.sentences
#     scans = [event.scan for event in block.scan_events]
#
#     stimuli = [event.stimulus_pointers for event in block.scan_events]
#     timestamps = [event.timestamp for event in block.scan_events]




# ---- HARRY POTTER DATA -----
# Get the data at: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/
# Make sure to change the dir!
#

# print("\n\nHarry Potter Data")
#reader = HarryPotterReader(data_dir=data_dir + "HarryPotter/")
#data = reader.read_all_events(subject_ids=[1, 2, 3, 4])

reader = PereiraReader(data_dir=data_dir + "pereira_data/subject_data/", experiment=3)
data = reader.read_all_events()

all_varied_voxels = []
colors = ["red", "yellow", "blue", "green", "orange", "white", "black", "brown", "magenta"]
color_data =[]
for subject in data.keys():
    scans = []
    roi_mapping = reader.get_voxel_to_region_mapping(subject)
    mni_mapping = reader.get_voxel_to_mni_mapping(subject)
    for block in data[subject]:
        scans.extend([event.scan for event in block.scan_events])
    scans = reduce_mean(scans)
    voxel_ids = select_most_varied_voxels(scans)[0:1000]


    varied_voxels = [mni_mapping[v] for v in voxel_ids]
    regions =  [roi_mapping[v] for v in voxel_ids]
    all_varied_voxels.extend(varied_voxels)
    color = colors[subject-1]
    color_data.extend([color for _ in range(len(varied_voxels))])
print(len(all_varied_voxels), len(color_data))
    # Open plot in browser
count_map = {}
for i in regions:
    count_map[str(i)] = count_map.get(str(i), 0) + 1
print(count_map)
view = plotting.view_markers(all_varied_voxels, color_data, marker_size=3)
view.open_in_browser()





# ---- KAPLAN DATA -----
# This dataset is described in Dehghani et al. 2017: https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23814
# I received it from Jonas Kaplan, but I am not allowed to share it.

# print("\n\n Stories Data")
# kaplan_reader = StoryDataReader(data_dir=data_dir + "Kaplan_data/")
# kaplan_data = kaplan_reader.read_all_events(subject_ids=[29],language="english")
# sum = 0
#
# for subject_id in kaplan_data.keys():
#     all_scans = []
#     for block in kaplan_data[subject_id]:
#         # These are all already sorted, so I think you don't even need timesteps.
#         sentences = block.sentences
#         scans = [event.scan for event in block.scan_events]
#         stimuli = [event.stimulus_pointers for event in block.scan_events]
#         timestamps = [event.timestamp for event in block.scan_events]
#         all_scans.append(scans[0])
#
# print("\n\nBLOCK: " + str(block.block_id))
# print("Number of scans: " + str(len(scans)))
# print("Number of sentences in story: " + str(len(sentences)))
# print("Number of timestamps: " + str(len(timestamps)))
# print("Example sentences 1-3: \n" + str(sentences[0:3]))
# print("Example stimuli 0-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[0:20]))


# The Stories data is also available in Farsi and Chinese
# farsi_story_data = readKaplanData.read_all(data_dir + "Kaplan_data", "farsi")
# chinese_story_data = readKaplanData.read_all(data_dir + "Kaplan_data", "chinese")


# ---- NBD DATA (Dutch) ----
# Get the data at: https://osf.io/utpdy/
# The NBD data is very big, start with a subset.
# # Make sure to change the dir:
# nbd_dir = data_dir + "NBD/"
# print("\n\nDutch Narrative Data")
# nbd_reader = NBDReader(data_dir=nbd_dir)
# nbd_data = nbd_reader.read_all_events()
# print("Number of scans: " + str(len(nbd_data)))
# print("Subjects: " + str({event.subject_id for event in nbd_data}))
# print("Runs: " + str({event.block for event in nbd_data}))
# print("Examples: ")
# for i in range(0, 10):
#     print(vars(nbd_data[i]))


# ---- Pereira DATA -----
# Make sure to get the data at https://evlab.mit.edu/sites/default/files/documents/index.html
#
# print("\n\Pereira Data")
# pereira_reader = PereiraReader(data_dir=data_dir + "pereira_data/", experiment=2)
#
# # for testing experiment 1
# # subject_id = [11, 2]
#
# # for testing experiment 2 & 3
# subject_id = [2, 4, 7]
#
# # general testing with 1 subject
# # subject_id = [2]
#
# pereira_data = pereira_reader.read_all_events(subject_ids=subject_id)
# all_scans = []
# for subject_id in subject_id:
#     for block in pereira_data[subject_id]:
#         sentences = block.sentences
#         scans = [event.scan for event in block.scan_events]
#         stimuli = [event.stimulus_pointers for event in block.scan_events]
#         timestamps = [event.timestamp for event in block.scan_events]


