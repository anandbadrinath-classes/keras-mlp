from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np
import colorama
from colorama import Fore

def exercise1():
    dm = DataManager()
    dm.preprocessData()
    outstring = ("Testing image preprocessing: ")
    if dm.train_data.dtype != np.float64:
        outstring += ("{}KO| Expected train_data type to be float64, got {}{}"
                      "".format(Fore.RED, dm.train_data.dtype, Fore.RESET))
    elif dm.eval_data is None:
        outstring += ("{}KO| Expected eval_data type to be float64, got {}{}"
                      "".format(Fore.RED, None, Fore.RESET))
    elif dm.eval_labels is None:
        outstring += ("{}KO| Expected eval_labels type to be float64, got {}{}"
                      "".format(Fore.RED, None, Fore.RESET))
    elif dm.eval_data.dtype !=np.float64:
        outstring += ("{}KO| Expected eval_data type to be float64, got {}{}"
                      "".format(Fore.RED, dm.train_data.dtype, Fore.RESET))
    elif dm.test_data.dtype != np.float64:
        outstring += ("{}KO| Expected test_data type to be float64, got {}{}"
                      "".format(Fore.RED, dm.train_data.dtype, Fore.RESET))
    elif (dm.train_data < 0).any() or (dm.train_data > 1).any():
        outstring += ("{}KO| Train data values should be between 0 and 1.{}"
                      "".format(Fore.RED, Fore.RESET))
    elif (dm.eval_data < 0).any() or (dm.eval_data > 1).any():
        outstring += ("{}KO| Train data values should be between 0 and 1.{}"
                      "".format(Fore.RED, Fore.RESET)) 
    elif (dm.test_data < 0).any() or (dm.test_data > 1).any():
        outstring += ("{}KO| Train data values should be between 0 and 1.{}"
                      "".format(Fore.RED, Fore.RESET))
    elif dm.eval_data.shape != (6000, 28, 28):
        outstring += ("{}KO| Expected eval_data length to be (6000, 28, 28), got {}{}"
                      "".format(Fore.RED, dm.eval_data.shape, Fore.RESET))
    elif dm.eval_labels.shape != (6000,):
        outstring += ("{}KO| Expected eval_labels length to be (6000,), got {}{}"
                      "".format(Fore.RED, dm.eval_labels.shape, Fore.RESET))
    elif dm.test_data.shape != (4000, 28, 28):
        outstring += ("{}KO| Expected test_data length to be (4000, 28, 28), got {}{}"
                      "".format(Fore.RED, dm.test_data.shape, Fore.RESET))
    elif dm.test_labels.shape != (4000,):
        outstring += ("{}KO| Expected test_labels length to be (4000,), got {}{}"
                      "".format(Fore.RED, dm.test_labels.shape, Fore.RESET))
    else:
        outstring += "{}OK{}".format(Fore.GREEN, Fore.RESET)

    print (outstring)
    
def testInputLayer(imageWidth, imageHeight):
    nn = NeuralNetwork()
    nn.setInputLayer(imageWidth,imageWidth)
    outstring = ("Testing input layer creation: ")
    if nn.inputLayer is None:
        outstring += ("{}KO| inputLayer was not set{}"
                      "".format(Fore.RED, Fore.RESET))
        return outstring
    config = nn.inputLayer.get_config()
    if "flatten" not in config["name"]:
        outstring += ("{}KO| Expected inputLayer to be of type flatten but got {}{}"
                      "".format(Fore.RED, config["name"].split("_")[0], Fore.RESET))
        return outstring
    if config["batch_input_shape"] != (None, imageWidth, imageHeight):
        outstring += ("{}KO| Expected inputLayer shape to be {} but got {}{}"
                      "".format(Fore.RED, (None, imageWidth, imageHeight),
                                config["batch_input_shape"], Fore.RESET))
        return outstring

    outstring += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outstring

def exercise2():
    print (testInputLayer(1,1))
    print (testInputLayer(2,2))
    print (testInputLayer(28,28))


def testHiddenLayer(numberOfNeurons):
    nn = NeuralNetwork()
    nn.setHiddenLayer(numberOfNeurons)
    outstring = ("Testing hidden layer creation: ")
    if nn.hiddenLayer is None:
        outstring += ("{}KO| hiddenLayer was not set{}"
                      "".format(Fore.RED, Fore.RESET))
        return outstring
    config = nn.hiddenLayer.get_config()
    if "dense" not in config["name"]:
        outstring += ("{}KO| Expected hiddenLayer to be of type dense but got {}{}"
                      "".format(Fore.RED, config["name"].split("_")[0], Fore.RESET))
        return outstring
    if config["units"] != numberOfNeurons:
        outstring += ("{}KO| Expected hiddenLayer to have {} neurons but got {}{}"
                      "".format(Fore.RED, numberOfNeurons,
                                config["units"], Fore.RESET))
        return outstring


    outstring += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outstring

def exercise3():
    print (testHiddenLayer(1))
    print (testHiddenLayer(2))
    print (testHiddenLayer(128))
    print (testHiddenLayer(512))

def testOutputLayer(numberOfNeurons):
    nn = NeuralNetwork()
    nn.setOutputLayer(numberOfNeurons)
    outstring = ("Testing output layer creation: ")
    if nn.outputLayer is None:
        outstring += ("{}KO| outputLayer was not set{}"
                      "".format(Fore.RED, Fore.RESET))
        return outstring
    config = nn.outputLayer.get_config()
    if "dense" not in config["name"]:
        outstring += ("{}KO| Expected outputLayer to be of type dense but got {}{}"
                      "".format(Fore.RED, config["name"].split("_")[0], Fore.RESET))
        return outstring
    if config["units"] != numberOfNeurons:
        outstring += ("{}KO| Expected outputLayer to have {} neurons but got {}{}"
                      "".format(Fore.RED, numberOfNeurons,
                                config["units"], Fore.RESET))
        return outstring


    outstring += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outstring

def exercise4():
    print (testHiddenLayer(1))
    print (testHiddenLayer(10))
    print (testHiddenLayer(20))
    print (testHiddenLayer(40))

def exercise5():
    dm = DataManager()
    dm.preprocessData()
    nn = NeuralNetwork()
    nn.setInputLayer(28,28)
    nn.setHiddenLayer(28)
    nn.setOutputLayer(10)
    nn.createModel()
    print("Testing train function: ")
    nn.train(dm.train_data, dm.train_labels, 10)

def exercise6():
    dm = DataManager()
    dm.preprocessData()
    nn = NeuralNetwork()
    nn.setInputLayer(28,28)
    nn.setHiddenLayer(28)
    nn.setOutputLayer(10)
    nn.createModel()
    nn.train(dm.train_data, dm.train_labels, 1)
    outstring = "Testing eval function: "
    eval_results = nn.evaluate(dm.eval_data, dm.eval_labels)
    if eval_results is None:
        outstring += ("{}KO| no results were returned{}"
                      "".format(Fore.RED, Fore.RESET))
    else:
       outstring += "{0}OK| Test accuracy was: {1:.4f}{2}".format(Fore.GREEN, 
                                                                  eval_results, 
                                                                  Fore.RESET)

    print (outstring)

def exercise7():
    dm = DataManager()
    dm.preprocessData()
    nn = NeuralNetwork()
    nn.setInputLayer(28,28)
    nn.setHiddenLayer(28)
    nn.setOutputLayer(10)
    nn.createModel()
    nn.train(dm.train_data, dm.train_labels, 1)
    nn.saveModel("test.h5")
    nn2 = NeuralNetwork()
    nn2.loadModel("test.h5")
    eval_results = nn.evaluate(dm.eval_data, dm.eval_labels)
    eval_results2 = nn2.evaluate(dm.eval_data, dm.eval_labels)
    outstring = "Testing save and load functions: "
    if eval_results != eval_results2:
        outstring += ("{}KO| evaluation results for loaded network were {} expected {}{}"
                      "".format(Fore.RED, eval_results2, eval_results, Fore.RESET))

    else:
        outstring += "{}OK{}".format(Fore.GREEN, Fore.RESET)

    print(outstring)

# Write your own end to end test, try to get the best evaluation results possible
def exercise8():
    pass
    

if __name__ == "__main__":
    colorama.init()
    # DataManager
    exercise1()
    
    # NeuralNetwork 
    # exercise2()
    # exercise3()
    # exercise4()
    # exercise5()
    # exercise6()
    # exercise7()
    # exercise8()