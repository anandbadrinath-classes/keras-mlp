import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import pickle

class DataManager(object):

    def __init__(self):
        self.eval_data = None
        self.eval_labels = None
        self.loadData()

    def loadData(self, fromPkl=False):
        if fromPkl:
            (trd, trl), (ted, tel) = pickle.load(open("mnist.pkl", "rb"))
        else:
            (trd, trl), (ted, tel) = mnist.load_data()
        self.train_data = trd
        self.train_labels = trl
        self.test_data = ted
        self.test_labels = tel

    # Exercise 1: Convert the image format
    def preprocessData(self):
        """Convert the train and test images from uint8 (0-255) to float64 
           (0.0 - 1.0), then split the test_data and labels into 2 sets:
                - eval which contains 60% of the data
                - test which contains the remaining 40%
        """
