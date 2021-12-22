from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from image_processing.image_manager import ImageManager
from multiprocessing import Process
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import numpy as np
import pickle
import time

class TrainingModule:
    def __init__(self, chosen_algorithm = "logistic_regression"):
        self.__log_reg = SGDClassifier(loss = "log")
        self.__mlp = MLPClassifier(activation = "logistic", verbose = True, random_state = 1, shuffle = True, warm_start = True)

        self.__chosen_algorithm = chosen_algorithm

        self.__im = ImageManager()
        self.__positive_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/1/"
        self.__negative_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/0/"

        # self.__positive_examples_count = self.__im.get_positive_examples_count()
        # self.__negative_examples_count = self.__im.get_negative_examples_count()

        self.__positive_examples_count = 78769
        self.__negative_examples_count = 78769

        # self.__positive_examples_count = 50000
        # self.__negative_examples_count = 50000

        # Training set contains 70% of all examples
        self.__training_set_positive_examples_count = int(0.7 * self.__positive_examples_count)
        self.__training_set_negative_examples_count = int(0.7 * self.__negative_examples_count)
        
        # Validation set contains 15% of all examples
        self.__validation_set_positive_examples_count = int((self.__positive_examples_count - self.__training_set_positive_examples_count) / 2)
        self.__validation_set_negative_examples_count = int((self.__negative_examples_count - self.__training_set_negative_examples_count) / 2)
        
        # Test set contains 15% of all examples
        self.__test_set_positive_examples_count = self.__positive_examples_count - (self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count)
        self.__test_set_negative_examples_count = self.__negative_examples_count - (self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count)

        self.__batch_size = 1000

        self.__attributes_count = 7500
        self.__checkpoint = 1000

        self.__finalized_model_filename = "finalized_model.sav"
        self.__output_dir = "./output"
        self.__saved_models_dir = "./saved_models"

        self.__classes = np.array([0, 1])

        self.__start_time = time.time()

    def set_algorithm(self, algorithm):
        if algorithm == "logistic_regression" or algorithm == "neural_network":
            self.__chosen_algorithm = algorithm 

    def __save_model(self):
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        model_name = self.__chosen_algorithm + "-" + str(current_time) + "_" + self.__finalized_model_filename
        if not os.path.isdir(self.__saved_models_dir):
            os.mkdir(self.__saved_models_dir)
        if self.__chosen_algorithm == "logistic_regression":
            pickle.dump(self.__log_reg, open(self.__saved_models_dir + "/" + model_name, "wb"))
        elif self.__chosen_algorithm == "neural_network":
            pickle.dump(self.__mlp, open(self.__saved_models_dir + "/" + model_name, "wb"))

    def __load_logistic_regression_model(self, path):
        self.__log_reg = pickle.load(open(path, "rb"))

    def __load_neural_network_model(self, path):
        self.__mlp = pickle.load(open(path, "rb"))

    def __output_results(self, filename, content):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        file_path = self.__output_dir + "/" + filename + "_" + self.__chosen_algorithm + "_" + str(current_time) + ".txt"
        with open(file_path, 'w+') as f:
            f.write(content)

    def run(self):
        self.__train()
        self.__test()

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

        seconds_passed = time.time() - self.__start_time
        print("[INFO] Program finished after", str(int(seconds_passed / 60)) + ":" + str(seconds_passed % 60), "minutes")

    def __train(self, learning_rate = "optimal", regularization = 0.0001, iterations = 1000, hidden_layer = (5000, 5000), solver = "sgd"):
        print("\n*--------------------TRAINING--------------------*")
        print("[INFO] Learning from", self.__training_set_positive_examples_count + self.__training_set_negative_examples_count, "images in total.")
        print("[INFO] Initializing model.")

        # Reinitialization of the classifier object
        if self.__chosen_algorithm == "logistic_regression":
            self.__log_reg = SGDClassifier(loss = "log", shuffle = True, verbose = 1, eta0 = 1.0, warm_start = True)
            self.__log_reg.learning_rate = learning_rate
            self.__log_reg.alpha = regularization
            self.__log_reg.max_iter = iterations
        elif self.__chosen_algorithm == "neural_network":
            self.__mlp = MLPClassifier(activation = "logistic", verbose = True, random_state = 1, shuffle = True, warm_start = True)
            self.__mlp.hidden_layer_sizes = hidden_layer
            self.__mlp.solver = solver
            self.__mlp.alpha = regularization
            self.__mlp.max_iter = iterations

        # Training
        print("[INFO] Training of", self.__chosen_algorithm, "model started.")

        positive_images_processed = 0
        negative_images_processed = 0

        while positive_images_processed < self.__training_set_positive_examples_count or negative_images_processed < self.__training_set_negative_examples_count:
            positive_images_left = self.__training_set_positive_examples_count - positive_images_processed
            negative_images_left = self.__training_set_negative_examples_count - negative_images_processed

            # Creating a batch of mixed positive and negative examples (size of a batch is preserved)            
            positive_images_batch_count = int(self.__batch_size / 2)
            negative_images_batch_count = self.__batch_size - positive_images_batch_count

            if positive_images_left == 0 and negative_images_left > 0:
                negative_images_batch_count = self.__batch_size

            if negative_images_left == 0 and positive_images_left > 0:
                positive_images_batch_count = self.__batch_size

            if positive_images_left < positive_images_batch_count:
                positive_images_batch_count = positive_images_left

            if negative_images_left < negative_images_batch_count:
                negative_images_batch_count = negative_images_left

            if positive_images_batch_count < int(self.__batch_size / 2) and negative_images_left >= self.__batch_size - positive_images_batch_count:
                negative_images_batch_count = self.__batch_size - positive_images_batch_count

            if negative_images_batch_count < int(self.__batch_size / 2) and positive_images_left >= self.__batch_size - negative_images_batch_count:
                positive_images_batch_count = self.__batch_size - negative_images_batch_count

            first_positive_image = positive_images_processed + 1
            last_positive_image = first_positive_image + positive_images_batch_count
            first_negative_image = negative_images_processed + 1
            last_negative_image = first_negative_image + negative_images_batch_count

            # Load matrices of data
            (X_pos, y_pos) = self.__im.load_image_data(first_positive_image, last_positive_image, 1)
            (X_neg, y_neg) = self.__im.load_image_data(first_negative_image, last_negative_image, 0)

            X_training = np.vstack([X_pos, X_neg])
            y_training = np.hstack([y_pos, y_neg])

            # Shuffling data
            A = np.c_[X_training, y_training]
            np.random.shuffle(A)
            num_cols = np.shape(A)[1]

            X_training = A[:, 0:num_cols - 1]
            y_training = A[:, num_cols - 1]

            # Learning from batch
            if self.__chosen_algorithm == "logistic_regression":
                self.__log_reg.fit(X_training, y_training)
            elif self.__chosen_algorithm == "neural_network":
                self.__mlp.fit(X_training, y_training)

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")

        print("Total images processed:", positive_images_processed + negative_images_processed)
        print("[INFO] Training done.")

        
    def __validate(self):
        print("\n*--------------------VALIDATING--------------------*")
        print("[INFO] Validating on", self.__validation_set_positive_examples_count + self.__validation_set_negative_examples_count, "images in total.")

        # Validation
        print("[INFO] Validation of", self.__chosen_algorithm, "model started.")

        first_positive_image = self.__training_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__validation_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__validation_set_negative_examples_count

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data(first_negative_image, last_negative_image, 0)

        X_cv = np.vstack([X_pos, X_neg])
        y_cv = np.hstack([y_pos, y_neg])

        # Testing on a batch
        if self.__chosen_algorithm == "logistic_regression":
            accuracy = self.__log_reg.score(X_cv, y_cv)
            y_predicted = self.__log_reg.predict(X_cv)
        elif self.__chosen_algorithm == "neural_network":
            accuracy = self.__mlp.score(X_cv, y_cv)
            y_predicted = self.__mlp.predict(X_cv) 
        f1 = f1_score(y_cv, y_predicted)
        print("[INFO] Validation done.")

        return (accuracy, f1)

    def __test(self):
        print("\n*--------------------TESTING--------------------*")
        print("[INFO] Testing on", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "images in total.")

        # Testing
        print("[INFO] Testing of", self.__chosen_algorithm, "model started.")

        first_positive_image = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__test_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__test_set_negative_examples_count

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data(first_negative_image, last_negative_image, 0)

        X_test = np.vstack([X_pos, X_neg])
        y_test = np.hstack([y_pos, y_neg])

        # Testing on a batch
        if self.__chosen_algorithm == "logistic_regression":
            accuracy = self.__log_reg.score(X_test, y_test)
            y_predicted = self.__log_reg.predict(X_test)
        elif self.__chosen_algorithm == "neural_network":
            accuracy = self.__mlp.score(X_test, y_test)
            y_predicted = self.__mlp.predict(X_test)
        f1 = f1_score(y_test, y_predicted)

        print("Tested on a batch of", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "examples --->", self.__test_set_positive_examples_count, "positive and", self.__test_set_negative_examples_count, "negative")
        print("[INFO] Testing done.")

        print("\n*--------------------RESULTS--------------------*")
        print("[INFO] Chosen model:", self.__chosen_algorithm)
        print("[INFO] Accuracy obtained on the test set:", accuracy)
        print("[INFO] F1 score obtained on the test set:", f1)

        # Save output to file
        filename = self.__chosen_algorithm + "_results.txt" 
        content = "*----RESULTS----*\n\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        self.__output_results(filename, content)

        file_path = self.__output_dir + "_" + self.__chosen_algorithm + "_classifications.txt"
        f = open(file_path, 'w+')
        f.write("PRED     REAL\n")
        for i in range(0, y_test.shape[0]):
           f.write(str(y_predicted[i]) + "    " + str(y_test[i]) + "\n")
        f.close()
