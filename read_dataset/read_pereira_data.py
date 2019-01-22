import numpy as np
import scipy.io
from .scan_elements import Block, ScanEvent
from read_dataset.abstract_fmri_reader import FmriReader
from os import listdir
import os.path
import pickle
import spacy
import nltk


# =============================================================================
#   participant P01 has quite substantial amount of missing data; 
#   probably didn't matter for the authors since they selected the 5000 best voxels
# TODO How were the best voxells determined
# =============================================================================


# =============================================================================
# # there are also scans for an entire passage. These are not yet added as scan events, 
# # since it is not clear yet how they were computed (they do not equal the mean of the voxels of the sentences in the passage)
# # these activations would be necessary when looking at pairwise comparison (ii): different passages within the same topic 
# # They are not used on comparison (i): different sentences from different topics; and (iii): different sentences from the same passage
# =============================================================================


# This class reads the word and sentence data from Pereira et al., 2018
# Paper: https://www.nature.com/articles/s41467-018-03068-4
# Data: https://evlab.mit.edu/sites/default/files/documents/index.html
# Make sure to also check the supplementary material.


class PereiraReader(FmriReader):

    def __init__(self, data_dir, experiment=1, paradigm=1):
        super(PereiraReader, self).__init__(data_dir)
        self.paradigm = paradigm
        self.experiment = experiment
        self.topic_lookup = {}

    def read_all_events(self, subject_ids=None):

        # Collect scan events
        blocks = {}

        if subject_ids == None:
            subject_ids = [file for file in listdir(self.data_dir)]
        if self.experiment not in [1, 2, 3]:
            raise ValueError("Please choose experiment 1, 2, or 3")
        elif self.experiment == 1:
            output = self.read_experiment_1(subject_ids, blocks)  # word
            return output
        else:
            output = self.read_experiment_23(subject_ids, blocks)  # sentences
            return output

    def read_experiment_1(self, subject_ids, blocks):


        for subject_id in subject_ids:
            blocks_for_subject = []

            # possible paradigms
            paradigms = ["sentences", "pictures", "wordclouds", "average"]
            paradigm_index = self.paradigm - 1
            if paradigm_index not in range(len(paradigms)):
                raise ValueError(
                    "please fill in 1 (for sentences), 2 (for pictures), 3 (for clouds), or 4 (for average)")

            # make blocks for sentences, pictures, or clouds
            elif paradigm_index in [0, 1, 2]:
                if subject_id == "P01":
                    datafile = scipy.io.loadmat(
                        self.data_dir + subject_id + "/data_180concepts_" + paradigms[paradigm_index] + ".mat")
                else:
                    datafile = scipy.io.loadmat(
                        self.data_dir + str(subject_id) + "/data_180concepts_" + paradigms[paradigm_index] + ".mat")

                # time stamp 
                n = 1

                for i in range(0, len(datafile["examples"])):
                    scan = datafile["examples"][i]
                    word = datafile["keyConcept"][i][0][0]
                    scan_event = ScanEvent(subject_id, [(0, 0)], n, scan)
                    block = Block(subject_id, n, [[word]], [scan_event], None)
                    blocks_for_subject.append(block)
                    n += 1
                blocks[subject_id] = blocks_for_subject

                # make blocks for average
            else:
                all_scans = []
                words = []

                # collect scans of all paradigms
                for paradigm in range(paradigm_index):
                    if subject_id == "P01":
                        datafile = scipy.io.loadmat(
                            self.data_dir + subject_id + "/data_180concepts_" + paradigms[paradigm_index] + ".mat")
                    else:
                        datafile = scipy.io.loadmat(
                            self.data_dir + subject_id + "/data_180concepts_" + paradigms[paradigm] + ".mat")
                    all_scans.append([])
                    for i in range(0, len(datafile["examples"])):
                        scan = datafile["examples"][i]
                        all_scans[paradigm].append(scan)

                        # make word list (remains the same over paradigms)
                        if paradigm == 0:
                            word = datafile["keyConcept"][i][0][0]
                            words.append(word)

                            # average voxel values over all paradigms
                mean_scan = np.mean(all_scans, axis=0)

                # time stamp increments with 1 for words 
                n = 1

                for i in range(0, len(words)):
                    scan_event = ScanEvent(subject_id, [(0, 0)], n, mean_scan[i])
                    block = Block(subject_id, n, [[words[i]]], [scan_event], None)
                    blocks_for_subject.append(block)
                    n += 1
                blocks[subject_id] = blocks_for_subject

        return blocks

    def read_experiment_23(self, subject_ids, blocks):

        # the data sets all provide the stimuli (with belonging scans) in the same order. For scan times, I have now created an index that 
        # provides the time stamp of the scans in the order of the data file. This is not necessarily the order in which participants have seen them
        # I didn't deem this necessary, since after each sentence presentation (4s) there was a 4s fixation point. 
        # Judging by the presentation script on https://osf.io/crwz7/wiki/home/, for experiment 2 and 3, passages were 
        # pseudorandomly assigned to runs and randomly ordered within runs
        # A file with the explicit order of sentences per participant is not given

        # not all participants performed all experiments
        participants_exp2 = [2, 4, 7, 8, 9, 14, 15, "P01"]
        participants_exp3 = [2, 3, 4, 7, 15, "P01"]
        if subject_ids is None:
            if self.experiment == 2:
                subject_ids = participants_exp2
            else:
                subject_ids = participants_exp3

        # collect blocks for sujects
        for subject_id in subject_ids:
            blocks_for_subject = []

            if self.experiment == 2:

                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_384sentences")

            else:

                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_243sentences")

            all_passages = []
            store_passage_id = []
            sentence_pointer = 0
            all_events = []
            scan_time = 0

            # collect scans and (tokenized) sentences
            for i in range(len(datafile["examples_passagesentences"])):
                stimulus_pointer = []
                scan = datafile["examples_passagesentences"][i]
                sentence = str(datafile["keySentences"][i][0][0])
                tokenized_sentence = nltk.word_tokenize(sentence)

                # place sentences in passages
                # experiment 2 and 3 have put their data into a column vs. a row in "labelsPassageForEachSentence"
                if self.experiment == 2:
                    passage_id = datafile["labelsPassageForEachSentence"][i][0] - 1
                else:
                    passage_id = datafile["labelsPassageForEachSentence"][0][i] - 1
                if not passage_id in store_passage_id:
                    all_passages.append([])
                    sentence_pointer = 0
                    all_events.append([])
                    store_passage_id.append(passage_id)
                else:
                    sentence_pointer += 1
                all_passages[passage_id].append(tokenized_sentence)

                # make stimulus pointer
                for token_id in range(len(tokenized_sentence)):
                    stimulus_pointer.append((sentence_pointer, token_id))

                scan_time += 1

                # make event per scan and place in passages
                event = ScanEvent(subject_id, stimulus_pointer, scan_time, scan)
                all_events[passage_id].append(event)

            for passage_id in range(len(store_passage_id)):
                # form blocks
                block = Block(str(subject_id), passage_id + 1, all_passages[passage_id], all_events[passage_id])
                blocks_for_subject.append(block)

                # make dictionary with topic for each passage
                topic_index = datafile["labelsPassageCategory"][passage_id][0] - 1
                topic = datafile["keyPassageCategory"][0][topic_index][0]
                self.topic_lookup[passage_id] = topic

            blocks[subject_id] = blocks_for_subject

        return blocks
    def get_voxel_to_xyz_mapping(self, subject_id):
        metadata = scipy.io.loadmat(  self.data_dir + subject_id + "/data_180concepts_sentences.mat")["meta"]
        coordinates_of_nth_voxel = metadata[0][0][6]

        voxel_to_xyz = {}
        for voxel in range(0, coordinates_of_nth_voxel.shape[0]):

            voxel_to_xyz[voxel] = coordinates_of_nth_voxel[voxel]
        #for name in roi_names:
         # print(name[0])
        return voxel_to_xyz