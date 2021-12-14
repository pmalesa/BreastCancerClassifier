from sklearn.linear_model import SGDClassifier
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
    def __init__(self):
        self.__log_reg = SGDClassifier(loss = "log")
        self.__im = ImageManager()
        self.__positive_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/1/"
        self.__negative_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/0/"
        self.__positive_examples_count = self.__im.get_positive_examples_count()
        self.__negative_examples_count = self.__im.get_negative_examples_count()

        # self.__positive_examples_count = 100
        # self.__negative_examples_count = 100

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

        self.__finalized_model_filename = "finalized_model_test.sav"
        self.__output_dir = "./output"
        self.__saved_models_dir = "./saved_models"

        self.__log_reg_classes = np.array([0, 1])

        self.__start_time = time.time()

    def __predict_class(self, X_new):
        y_prob = self.__log_reg.predict_proba(X_new)
        if y_prob[:, 1] >= 0.5:
            return 1
        else:
            return 0

    def __save_model(self):
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        model_name = str(current_time) + "_" + self.__finalized_model_filename
        if not os.path.isdir(self.__saved_models_dir):
            os.mkdir(self.__saved_models_dir)
        pickle.dump(self, open(self.__saved_models_dir + "/" + model_name, "wb"))

    def run(self):
        self.__train_logistic_regression_model()
        self.__save_model()
        self.__test_logistic_regression()

        # nIterations = 1000
        # regularization_values = [ 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 ]
        # solvers = ["sag", "saga", "lbfgs"]

        # f1_best = 0.0 
        # reg_best = 0.0
        # solver_best = ""
        # accuracy_best = 0.0   

        # for solver in solvers:
        #     for reg in regularization_values:
        #         # Training
        #         self.__train_logistic_regression_model(solver, reg, nIterations)
                
        #         # Validating
        #         (accuracy, f1) = self.__validate_logistic_regression()

        #         # Save output to file
        #         filename = "validation-" + self.__log_reg.solver + "-" + str(self.__log_reg.C) + "-" + str(self.__log_reg.max_iter)
        #         content = "*----VALIDATION----*\nSolver: " + self.__log_reg.solver + "\nregularization: " + str(self.__log_reg.C) + "\niterations: " + str(self.__log_reg.max_iter) + "\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        #         self.__output_results(filename, content)

        #         if f1 > f1_best:
        #             f1_best = f1
        #             accuracy_best = accuracy
        #             solver_best = solver
        #             reg_best = reg

        # print("*----BEST----*")
        # print("f1:", f1_best)
        # print("accuracy:", accuracy_best) 
        # print("solver:", solver_best)
        # print("regularization:", reg_best)

        # # Save output to file
        # filename = "best-" + solver_best + "-" + str(reg_best) + "-" + str(self.__log_reg.max_iter)
        # content = "*----BEST----*\nSolver: " + solver_best + "\nregularization: " + str(reg_best) + "\niterations: " + str(self.__log_reg.max_iter) + "\nF1: " + str(f1_best) + "\naccuracy: " + str(accuracy_best)
        # self.__output_results(filename, content)

        # # Loading previously trained model
        # # self.__load_model("finalized_model_test.sav")
        
        # # Testing
        # self.__train_logistic_regression_model(solver_best, reg_best, nIterations)
        # self.__test_logistic_regression()

        seconds_passed = time.time() - self.__start_time
        print("[INFO] Program finished after", str(int(seconds_passed / 60)) + ":" + str(seconds_passed % 60), "minutes")

    def __train_logistic_regression_model(self, learning_rate = "optimal", regularization = 0.0001, iterations = 1000):
        # Reinitialization of the classifier object
        self.__log_reg = SGDClassifier(loss = "log")
        self.__log_reg.learning_rate = learning_rate
        self.__log_reg.alpha = regularization
        self.__log_reg.max_iter = iterations

        print("\n*--------------------TRAINING--------------------*")
        print("[INFO] Learning from", self.__training_set_positive_examples_count + self.__training_set_negative_examples_count, "images in total.")

        # Training
        print("[INFO] Training started.")

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
            self.__log_reg.partial_fit(X_training, y_training, classes = self.__log_reg_classes)

            print("Learned from a batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative")

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count
            
        print("Total images processed:", positive_images_processed + negative_images_processed)
        print("[INFO] Training done.")

    def __load_model(self, path):
        pickle.load(open(path, "rb"))
        
    def __validate_logistic_regression(self):
        print("\n*--------------------VALIDATING--------------------*")
        print("[INFO] Validating on", self.__validation_set_positive_examples_count + self.__validation_set_negative_examples_count, "images in total.")

        # Shuffling data
        # A = np.c_[self.__X_cv, self.__y_cv]
        # np.random.shuffle(A)
        # num_cols = np.shape(A)[1]

        # self.__X_cv = A[:, 0:num_cols - 1]
        # self.__y_cv = A[:, num_cols - 1]

        # Validation
        print("[INFO] Validation started.")
        accuracy = self.__log_reg.score(self.__X_cv, self.__y_cv)
        y_predicted = np.zeros(self.__y_cv.shape, dtype = float)
        i = 0
        for example in self.__X_cv:
            example = example[np.newaxis, :]
            y_predicted[i] = self.__log_reg.predict(example)
            i += 1
        f1 = f1_score(self.__y_cv, y_predicted) 
        print("[INFO] Validation done.")

        # Save output to file
        filename = "validation-" + self.__log_reg.solver + "-" + str(self.__log_reg.C) + "-" + str(self.__log_reg.max_iter)
        content = "*----VALIDATION----*\nSolver: " + self.__log_reg.solver + "\nregularization: " + str(self.__log_reg.C) + "\niterations: " + str(self.__log_reg.max_iter)
        self.__output_results(filename, content)

        return (accuracy, f1)

    def __test_logistic_regression(self):
        print("\n*--------------------TESTING--------------------*")
        print("[INFO] Testing on", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "images in total.")

        # Testing
        print("[INFO] Testing started.")
        positive_images_processed = 0
        negative_images_processed = 0

        accuracy_scores = np.array([])
        # f1_scores = np.array([])

        while positive_images_processed < self.__test_set_positive_examples_count or negative_images_processed < self.__test_set_negative_examples_count:
            positive_images_left = self.__test_set_positive_examples_count - positive_images_processed
            negative_images_left = self.__test_set_negative_examples_count - negative_images_processed

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

            X_test = np.vstack([X_pos, X_neg])
            y_test = np.hstack([y_pos, y_neg])

            # Testing on a batch
            accuracy = self.__log_reg.score(X_test, y_test)
            y_predicted = np.zeros(y_test.shape, dtype = float)
            i = 0
            for example in X_test:
                example = example[np.newaxis, :]
                y_predicted[i] = self.__log_reg.predict(example)
                i += 1
            # f1 = f1_score(y_test, y_predicted)

            accuracy_scores = np.append(accuracy_scores, accuracy)
            # np.append(f1_scores, f1)

            print("Tested on a batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative")

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

        print("[INFO] Testing done.")

        accuracy = np.average(accuracy_scores)
        # f1 = np.average(f1_scores)

        print("\n*--------------------RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy * 100, "%")
        # print("[INFO] F1 score obtained on the test set:", f1)

    def __output_results(self, filename, content):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        file_path = self.__output_dir + "/" + filename + "_" + str(current_time) + ".txt"
        with open(file_path, 'w+') as f:
            f.write(content)