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










        # -----------------------------------------------------------------------------------------------------------------------------

        # if self.__chosen_algorithm == "logistic_regression":
        #     iterations = 1000
        #     regularization_values = [ 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0 ]
        #     learning_rates = ["optimal", "constant", "invscaling", "adaptive"]

        #     f1_best = 0.0 
        #     reg_best = 0.0
        #     learning_rate_best = ""
        #     accuracy_best = 0.0   

        #     for learning_rate in learning_rates:
        #         for reg in regularization_values:
        #             # Training
        #             print("Training with:", str(reg), str(learning_rate))
        #             self.__train(learning_rate = learning_rate, regularization = reg, iterations = iterations)
                    
        #             # Validating
        #             (accuracy, f1) = self.__validate()

        #             # Save output to file
        #             filename = "validation-" + learning_rate + "-" + str(reg) + "-" + str(iterations)
        #             content = "*----VALIDATION----*\n\nLearning rate: " + learning_rate + "\nregularization: " + str(reg) + "\niterations: " + str(iterations) + "\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        #             self.__output_results(filename, content)

        #             if f1 > f1_best:
        #                 f1_best = f1
        #                 accuracy_best = accuracy
        #                 learning_rate_best = learning_rate
        #                 reg_best = reg

        #     print("*----BEST----*")
        #     print("f1:", f1_best)
        #     print("accuracy:", accuracy_best) 
        #     print("learning rate:", learning_rate_best)
        #     print("regularization:", reg_best)

        #     # Save output to file
        #     filename = "validation-best-" + learning_rate_best + "-" + str(reg_best) + "-" + str(self.__log_reg.max_iter)
        #     content = "*----VALIDATION-BEST----*\n\nLearning rate: " + learning_rate_best + "\nregularization: " + str(reg_best) + "\niterations: " + str(iterations) + "\nF1: " + str(f1_best) + "\naccuracy: " + str(accuracy_best)
        #     self.__output_results(filename, content)

        # elif self.__chosen_algorithm == "neural_network":
        #     iterations = 300
        #     solvers = [ "sgd", "adam" ]
        #     regularization_values = [ 0.0001, 0.005, 0.01, 0.05, 0.1, 0.5 ]
        #     hidden_layers = [ (1000,), (5000,), (1000, 1000), (5000, 5000)]

        #     f1_best = 0.0 
        #     reg_best = 0.0
        #     solver_best = ""
        #     hidden_layer_best = []
        #     accuracy_best = 0.0  

        #     for solver in solvers:
        #         for hidden_layer in hidden_layers:
        #             for reg in regularization_values:
        #                 # Training
        #                 print("\nTraining with:", str(reg), str(hidden_layer), solver)
        #                 self.__train(regularization = reg, hidden_layer = hidden_layer, solver = solver, iterations = iterations)
                        
        #                 # Validating
        #                 (accuracy, f1) = self.__validate()

        #                 # Save output to file
        #                 filename = "validation-neural-network-" + solver_best + "-" + str(reg) + "-" + str(iterations)
        #                 content = "*----VALIDATION----*\n\nSolver: " + solver + "\nhidden layer: " + str(hidden_layer) + "\nregularization: " + str(reg) + "\niterations: " + str(iterations) + "\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        #                 self.__output_results(filename, content)

        #                 if f1 > f1_best:
        #                     f1_best = f1
        #                     accuracy_best = accuracy
        #                     solver_best = solver
        #                     reg_best = reg
        #                     hidden_layer_best = hidden_layer

        #     # Save output to file
        #     filename = "validation-best-" + solver_best + "-" + str(reg) + "-" + str(iterations)
        #     content = "*----VALIDATION-BEST---*\n\nSolver best: " + solver_best + "\nhidden layer best: " + str(hidden_layer_best) + "\nregularization best: " + str(reg_best) + "\niterations: " + str(iterations) + "\nF1: " + str(f1_best) + "\naccuracy: " + str(accuracy_best)
        #     self.__output_results(filename, content)

        # # Loading previously trained model
        # # self.__load_model("finalized_model_test.sav")
        
        # # Testing
        # if self.__chosen_algorithm == "logistic_regression":
        #     self.__train(learning_rate = learning_rate_best, regularization = reg_best, iterations = iterations)
        # elif self.__chosen_algorithm == "neural_network":    
        #     self.__train(regularization = reg_best, hidden_layer = hidden_layer_best, solver = solver_best, iterations = iterations)
        # self.__save_model()
        # self.__test()
        # -----------------------------------------------------------------------------------------------------------------------------
