import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

class NeuralNetwork(object):

    ## Exercise 2 Create the input layer of the model
    def setInputLayer(self, imageWidth, imageHeight):
        self.inputLayer = None

    ## Exercise 3 Create the hidden layer of the model
    def setHiddenLayer(self, numberOfNeurons):
        self.hiddenLayer = None

    ## Exercise 4 Create the output layer of the model
    def setOutputLayer(self, numberOfNeurons):
        self.outputLayer = None
    
    def createModel(self):
        """Creates and compiles a keras model from the input, hidden and output,
           layers
        """

        self.model = keras.Sequential([
                self.inputLayer,
                self.hiddenLayer,
                self.outputLayer
        ])

        self.model.compile(optimizer=keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    ## Exercise 5 Train the model using the keras.models API and return the 
    #  accuracy
    def train(self, train_data, train_labels, epochs):
        pass

    ## Exercise 6 Evaluate the model using the keras.models API and return the 
    #  accuracy 
    def evaluate(self, eval_data, eval_labels):
        pass

    def test(self, test_data):
        predictions = self.model.predict(test_data)
        return predictions

    ## Exercise 7 Save and load a model using the keras.models API
    def saveModel(self, saveFile="model.h5"):
        pass

    def loadModel(self, saveFile="model.h5"):
        pass