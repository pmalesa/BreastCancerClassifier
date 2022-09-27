from training.models.logistic_regression import LogististicRegression
from training.models.multi_layer_perceptron import MultiLayerPerceptron
from training.models.convolutional_neural_network import ConvolutionalNeuralNetwork
from training.models.support_vector_machine import SupportVectorMachine

from sklearn.metrics import f1_score

from image_processing.image_manager import ImageManager
from multiprocessing import Process
from datetime import datetime

import os
import numpy as np
import time

class TrainingModule:
    def __init__(self, chosen_algorithm = "logistic_regression"):
        self.__log_reg = LogististicRegression()
        self.__mlp = MultiLayerPerceptron()
        self.__svm = SupportVectorMachine()
        self.__cnn = ConvolutionalNeuralNetwork()

        self.__chosen_algorithm = chosen_algorithm

        self.__output_dir = "./output"
        self.__saved_models_dir = "./saved_models" 

    def set_algorithm(self, algorithm):
        if algorithm == "logistic_regression" or algorithm == "mlp" or algorithm == "cnn" or algorithm == "svm":
            self.__chosen_algorithm = algorithm 

    def save_chosen_model(self):
        if not os.path.isdir(self.__saved_models_dir):
            os.mkdir(self.__saved_models_dir)

        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        filename = self.__chosen_algorithm + "_" + current_time + ".sav"

        if self.__chosen_algorithm == "logistic_regression":
            self.__log_reg.save_model(self.__saved_models_dir)
        elif self.__chosen_algorithm == "mlp":
            self.__mlp.save_model(self.__saved_models_dir)
        elif self.__chosen_algorithm == "cnn":
            self.__cnn.save_model(self.__saved_models_dir)
        elif self.__chosen_algorithm == "svm":
            self.__svm.save_model(self.__saved_models_dir)

    def load_logistic_regression_model(self, path):
        self.__log_reg.load_model(path)

    def load_mlp_model(self, path):
        self.__mlp.load_model(path)

    def load_cnn_model(self, path):
        self.__cnn.load_model(path)

    def load_svm_model(self, path):
        self.__svm.load_model(path)

    def run(self):
        start_time = time.time()
        if self.__chosen_algorithm == "logistic_regression":
            self.__log_reg.start_learning_process()
        elif self.__chosen_algorithm == "mlp":
            self.__mlp.start_learning_process()
        elif self.__chosen_algorithm == "cnn":
            self.__cnn.start_learning_process()
        elif self.__chosen_algorithm == "svm":
            self.__svm.start_learning_process() 
        seconds_passed = time.time() - start_time
        print("[INFO] Program finished after", str(int(seconds_passed / 60)) + ":" + str(seconds_passed % 60), "minutes")

    def test(self):
        pass