"""
Utils script for the social LSTM implementation
Handles processing the input and target data in batches and sequences

Author : Anirudh Vemula
Date : 17th October 2016
"""

import os
import pickle
import random

import numpy as np


# The data loader class that loads data from the datasets considering
# each frame as a datapoint and a sequence of consecutive frames as the
# sequence.
class SocialDataLoader():

    def __init__(self, batch_size=5, input_seq_length=5, output_seq_length=1, maxNumPeds=70, datasets=None,
                 forcePreProcess=False, infer=False):
        """
        Initializer function for the SocialDataLoader class
        params:
        batch_size : Size of the mini-batch
        grid_size : Size of the social grid constructed
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        """
        # List of data directories where raw data resides
        print("************* SocialDataLoader is running *************")
        if datasets is None:
            datasets = [0, 1, 2, 3, 4]
        self.data_dirs = ['ucy/zara/zara01', 'ucy/zara/zara02', 'eth/univ', 'eth/hotel', 'ucy/univ']

        # Data directory where the pre-processed pickle file resides
        self.data_dir = '../data'

        self.used_data_dirs = [os.path.join(self.data_dir, self.data_dirs[x]) for x in datasets]
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Maximum number of peds in a single frame (Number obtained by checking the datasets)
        self.maxNumPeds = maxNumPeds

        # Store the arguments
        self.batch_size = batch_size
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

        # Validation arguments
        self.val_fraction = 0.2
        self.takeOneInNFrames = 6

        self.raw_data = None
        self.data = None
        self.frameList = None
        self.numPedsList = None
        self.valid_data = None

        self.dataset_pointer = 0
        self.frame_pointer = 0

        self.valid_dataset_pointer = 0
        self.valid_frame_pointer = 0

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not (os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def frame_preprocess(self, data_dirs, data_file):
        """
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        """

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, maxNumPeds, 3) where each pedestrian's
        # pedId, x, y , in each frame is stored
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:

            # Define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')
            file_path = os.path.join(directory, 'pixel_pos_interpolate.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            # Calculation of the number of frames, which however must be a number divisible by tekeOneInNFrames
            numFrames = int(len(frameList) / self.takeOneInNFrames) * self.takeOneInNFrames

            if self.infer:
                valid_numFrames = 0
            else:
                # Calculate the number of validation frames, as before this number must be divisible by
                # tekeOneInNFrames
                valid_numFrames = int((numFrames * self.val_fraction) / self.takeOneInNFrames) * self.takeOneInNFrames

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the numpy array for the current dataset
            # The size of the array must be divisible by tekeOneInNFrames
            all_frame_data.append(
                np.zeros((int((numFrames - valid_numFrames) / self.takeOneInNFrames), self.maxNumPeds, 3)))
            # Initialize the numpy array for the current dataset
            # As before, the size of the array must be divisible by tekeOneInNFrames
            valid_frame_data.append(np.zeros((int(valid_numFrames / self.takeOneInNFrames), self.maxNumPeds, 3)))

            # index to maintain the current frame
            ind = 0
            # loop that slides frames at long steps takeOneInNFrames
            while ind < numFrames:
                frame = frameList[ind]
                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()
                # if len(pedsList) > 0:
                #     print("frame#", frame, "\npedsList#", len(pedsList), " :", pedsList)

                # Helper print statement to figure out the maximum number of peds in any frame in any dataset
                # if len(pedsList) > 1:
                # print len(pedsList)
                # DEBUG
                #    continue

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y])

                if (ind >= valid_numFrames) or self.infer:
                    # Add the details of all the peds in the current frame to all_frame_data
                    # the index must be divided by takeOneInNFrames
                    all_frame_data[dataset_index][int((ind - valid_numFrames) / self.takeOneInNFrames), 0:len(pedsList),
                    :] = np.array(pedsWithPos)

                else:
                    # the index must be divided by takeOneInNFrames
                    valid_frame_data[dataset_index][int(ind / self.takeOneInNFrames), 0:len(pedsList), :] = np.array(
                        pedsWithPos)
                # Increase in the number of frames considered, must be increased by takeOneInNFrames
                ind += self.takeOneInNFrames
            # Increment the dataset index
            # print(all_frame_data, valid_frame_data)
            dataset_index += 1

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        """
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        """
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            # print('Training data from dataset', dataset, ':', len(all_frame_data))
            # print('Validation data from dataset', dataset, ':', len(valid_frame_data))
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.input_seq_length + 2)) - 1
            valid_counter += int(len(valid_frame_data) / (self.input_seq_length + 2)) - 1

        # Calculate the number of batches
        self.num_batches = int(counter / self.batch_size)
        self.valid_num_batches = int(valid_counter / self.batch_size)
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    def next_batch(self, valid=False, randomUpdate=False):
        """
        Function to get the next batch of points
        """
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            if valid:
                frame_data = self.valid_data[self.valid_dataset_pointer]
            else:
                frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            if valid:
                idx = self.valid_frame_pointer
            else:
                idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.input_seq_length + self.output_seq_length < frame_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = frame_data[idx:idx + self.input_seq_length + self.output_seq_length, :]

                seq_source_frame_data = frame_data[idx:idx + self.input_seq_length, :]
                seq_target_frame_data = frame_data[
                                        idx + self.output_seq_length:
                                        idx + self.input_seq_length + self.output_seq_length, :]
                # Number of unique peds in this sequence of frames
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                numUniquePeds = pedID_list.shape[0]

                sourceData = np.zeros((self.input_seq_length, self.maxNumPeds, 3))
                # targetData = np.zeros((self.output_seq_length, self.maxNumPeds, 3))
                targetData = np.zeros((self.input_seq_length, self.maxNumPeds, 3))
                for seq in range(self.input_seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    if numUniquePeds > self.maxNumPeds:
                        print("Max num peds surpassed: " + str(numUniquePeds) + " out of " + str(self.maxNumPeds))
                        numUniquePeds = self.maxNumPeds

                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]

                        if pedID == 0:
                            continue
                        else:
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            # print("\n\nseq#", seq, " ped# ", ped, "\n\tsped#{}: {}".format(sped.size, sped))
                            if sped.size != 0:
                                sourceData[seq, ped, :] = sped

                            tped = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])
                            # print("\n\ttped#{}: {}".format(tped.size, tped))
                            if tped.size != 0:
                                targetData[seq, ped, :] = tped
                # print("sourceData#{}: {}".format(sourceData.shape, sourceData),
                #       "\ntargetData#{}: {}".format(targetData.shape, targetData))

                x_batch.append(sourceData)
                y_batch.append(targetData)

                # Advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.input_seq_length + self.output_seq_length)
                else:
                    self.frame_pointer += (self.input_seq_length + self.output_seq_length)

                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self, valid=False):
        """
        Reset all pointers
        """
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
