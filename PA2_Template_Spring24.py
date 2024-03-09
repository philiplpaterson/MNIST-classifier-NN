import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import pickle


class My_dataset(Dataset):
    """
    Dataset Class for any dataset.
    This is a python class object, it inherits functions from 
    the pytorch Dataset object.
    For anyone unfamiliar with the python class object, see 
    https://www.w3schools.com/python/python_classes.asp
    or a more complicated but more detailed tutorial
    https://docs.python.org/3/tutorial/classes.html
    For anyone familiar with python class, but unfamiliar with pytorch
    Dataset object, see 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(self, data_dir, anno_csv) -> object:
        self.anno_data = pd.read_csv(anno_csv)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.anno_data)

    def __getitem__(self, idx):
        data_name = self.anno_data.iloc[idx, 0]
        data_location = self.data_dir + data_name
        data = np.float32(np.load(data_location))
        # This is for one-hot encoding of the output label
        gt_y = np.float32(np.zeros(10))
        index = self.anno_data.iloc[idx, 1]
        gt_y[index] = 1
        return data, gt_y


def PA2_train():
    # Specifying the training directory and label files
    train_dir = './'
    train_anno_file = './data_prog2Spring24/labels/train_anno.csv'

    # Specifying the device to GPU/CPU. Here, GPU means 'cuda' and CPU means 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the data and labels from the training data
    MNIST_training_dataset = My_dataset(data_dir=train_dir, anno_csv=train_anno_file)


    #You can set up your own maximum epoch. You may need  5 or 10 epochs to have a correct model.
    my_max_epoch = 10
    epochs = np.arange(0, my_max_epoch)


    for epoch in epochs:

        #Randomly split your training data into mini-batches where each mini-batch has 50 samples
        #Since we have 50000 training samples, and each batch has 50 samples,
        #the total number of batch will be 1000
        # YOU ARE NOT ALLOWED TO USE DATALOADER CLASS FOR RANDOM BATCH SELECTION
        total_batch = 1000

        for b in range(total_batch):
            '''Compute the loss for each batch, gradient of loss with respect to W, 
                    and update W accordingly.'''

        #Take the mean of all the mini-batch losses and denote it as your loss of the current epoch
        #Collect loss for each epoch and save the parameter Theta after each epoch


    # Plot the training loss vs accuracy
    # Visualize the final weight matrix
    # Save the final weight matrix

def PA2_test():
    # Specifying the training directory and label files
    test_dir = './'
    test_anno_file = './data_prog2Spring24/labels/test_anno.csv'
    feature_length = 784
    # Specifying the device to GPU/CPU. Here, GPU means 'cuda' and CPU means 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load the Weight Matrix that has been saved after training


    # Read the data and labels from the testing data
    MNIST_testing_dataset = My_dataset(data_dir=test_dir, anno_csv=test_anno_file)

    # Predict Y using X and updated W.

    # Calculate accuracy,


if __name__ == "__main__":
    PA2_train()
    PA2_test()
