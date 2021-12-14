from sklearn.linear_model import LogisticRegression
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
        self.__log_reg = LogisticRegression(max_iter = 1000)
        self.__im = ImageManager()
        self.__positive_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/1/"
        self.__negative_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/0/"
        # self.__positive_examples_count = self.__im.get_number_of_files(self.__positive_images_directory)
        # self.__negative_examples_count = self.__im.get_number_of_files(self.__negative_images_directory)
        self.__positive_examples_count = 5
        self.__negative_examples_count = 5

        # Training set contains 60% of all examples
        self.__training_set_positive_examples_count = int(0.6 * self.__positive_examples_count)
        self.__training_set_negative_examples_count = int(0.6 * self.__negative_examples_count)
        
        # Validation set contains 20% of all examples
        self.__validation_set_positive_examples_count = int((self.__positive_examples_count - self.__training_set_positive_examples_count) / 2)
        self.__validation_set_negative_examples_count = int((self.__negative_examples_count - self.__training_set_negative_examples_count) / 2)
        
        # Test set contains 20% of all examples
        self.__test_set_positive_examples_count = self.__positive_examples_count - (self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count)
        self.__test_set_negative_examples_count = self.__negative_examples_count - (self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count)

        self.__attributes_count = 7500
        self.__checkpoint = 10

        self.__finalized_model_filename = "finalized_model_test.sav"
        self.__output_dir = "./output"
        self.__saved_models_dir = "./saved_models"

        self.__start_time = time.time()

        # --------------------LOADING-TRAINING-DATA--------------------
        print("[INFO] Loading training data.")

        y_training_positive = np.ones(self.__training_set_positive_examples_count)
        y_training_negative = np.zeros(self.__training_set_negative_examples_count)

        # Load positive examples
        pixel_colors_matrix_pos = np.zeros((self.__training_set_positive_examples_count, self.__attributes_count), dtype = float)
        example_index = 0
        for filenumber in range(1, self.__training_set_positive_examples_count + 1):
            filepath = self.__positive_images_directory + str(filenumber) + ".png"
            pixel_colors_vec = self.__im.get_pixel_colors(filepath)
            pixel_colors_matrix_pos[example_index, :] = pixel_colors_vec
            example_index += 1
            if filenumber % self.__checkpoint == 0:
                print("[INFO] Loaded", filenumber, "positive training images.")

        # Load negative examples
        pixel_colors_matrix_neg = np.zeros((self.__training_set_negative_examples_count, self.__attributes_count), dtype = float)
        example_index = 0
        for filenumber in range(1, self.__training_set_negative_examples_count + 1):
            filepath = self.__negative_images_directory + str(filenumber) + ".png"
            pixel_colors_vec = self.__im.get_pixel_colors(filepath)
            pixel_colors_matrix_neg[example_index, :] = pixel_colors_vec
            example_index += 1
            if filenumber % self.__checkpoint == 0:
                print("[INFO] Loaded", filenumber, "negative training images.")    

        # Concatenate positive and negative matrices
        self.__X_training = np.vstack([pixel_colors_matrix_pos, pixel_colors_matrix_neg])
        self.__y_training = np.hstack([y_training_positive, y_training_negative])

        print("[INFO] Training data loaded.")

        # --------------------LOADING-VALIDATION-DATA--------------------
        print("[INFO] Loading validation data.")

        y_cv_positive = np.ones(self.__validation_set_positive_examples_count)
        y_cv_negative = np.zeros(self.__validation_set_negative_examples_count)

        first_positive_cv_image_number = self.__test_set_positive_examples_count + self.__training_set_positive_examples_count + 1
        first_negative_cv_image_number = self.__test_set_negative_examples_count + self.__training_set_negative_examples_count + 1

        # Load positive validation examples
        pixel_colors_matrix_pos = np.zeros((self.__validation_set_positive_examples_count, self.__attributes_count), dtype = float)
        example_index = 0
        for filenumber in range(first_positive_cv_image_number, first_positive_cv_image_number + self.__validation_set_positive_examples_count):
            filepath = self.__positive_images_directory + str(filenumber) + ".png"
            pixel_colors_vec = self.__im.get_pixel_colors(filepath)
            pixel_colors_matrix_pos[example_index, :] = pixel_colors_vec
            example_index += 1
            if filenumber % self.__checkpoint == 0:
                print("[INFO] Loaded", filenumber, "positive validation images.")

        # Load negative validation examples
        pixel_colors_matrix_neg = np.zeros((self.__validation_set_negative_examples_count, self.__attributes_count), dtype = float)
        example_index = 0
        for filenumber in range(first_negative_cv_image_number, first_negative_cv_image_number + self.__validation_set_negative_examples_count):
            filepath = self.__negative_images_directory + str(filenumber) + ".png"
            pixel_colors_vec = self.__im.get_pixel_colors(filepath)
            pixel_colors_matrix_neg[example_index, :] = pixel_colors_vec
            example_index += 1
            if filenumber % self.__checkpoint == 0:
                print("[INFO] Loaded", filenumber, "negative validation images.")

        # Concatenate positive and negative matrices
        self.__X_cv = np.vstack([pixel_colors_matrix_pos, pixel_colors_matrix_neg])
        self.__y_cv = np.hstack([y_cv_positive, y_cv_negative])

        print("[INFO] Validation data loaded.")

        # --------------------LOADING-TEST-DATA--------------------
        print("[INFO] Loading test data...")

        y_test_positive = np.ones(self.__test_set_positive_examples_count)
        y_test_negative = np.zeros(self.__test_set_negative_examples_count)

        first_positive_test_image_number = self.__training_set_positive_examples_count + 1
        first_negative_test_image_number = self.__training_set_negative_examples_count + 1

        # Load positive test examples
        pixel_colors_matrix_pos = np.zeros((self.__test_set_positive_examples_count, self.__attributes_count), dtype = float)
        example_index = 0
        for filenumber in range(first_positive_test_image_number, first_positive_test_image_number + self.__test_set_positive_examples_count):
            filepath = self.__positive_images_directory + str(filenumber) + ".png"
            pixel_colors_vec = self.__im.get_pixel_colors(filepath)
            pixel_colors_matrix_pos[example_index, :] = pixel_colors_vec
            example_index += 1
            if filenumber % self.__checkpoint == 0:
                print("[INFO] Loaded", filenumber, "positive test images.")

        # Load negative test examples
        pixel_colors_matrix_neg = np.zeros((self.__test_set_negative_examples_count, self.__attributes_count), dtype = float)
        example_index = 0
        for filenumber in range(first_negative_test_image_number, first_negative_test_image_number + self.__test_set_negative_examples_count):
            filepath = self.__negative_images_directory + str(filenumber) + ".png"
            pixel_colors_vec = self.__im.get_pixel_colors(filepath)
            pixel_colors_matrix_neg[example_index, :] = pixel_colors_vec
            example_index += 1
            if filenumber % self.__checkpoint == 0:
                print("[INFO] Loaded", filenumber, "negative test images.")

        # Concatenate positive and negative matrices
        self.__X_test = np.vstack([pixel_colors_matrix_pos, pixel_colors_matrix_neg])
        self.__y_test = np.hstack([y_test_positive, y_test_negative])

        print("[INFO] Test data loaded.")

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
        nIterations = 1000
        regularization_values = [ 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 ]
        solvers = ["sag", "saga"]

        f1_best = 0.0 
        reg_best = 0.0
        solver_best = ""
        accuracy_best = 0.0   

        for reg in regularization_values:
            for solver in solvers:
                # Training
                self.__train_logistic_regression_model(solver, reg, nIterations)
                
                # Validating
                (accuracy, f1) = self.__validate_logistic_regression()

                # Save output to file
                filename = "validation-" + self.__log_reg.solver + "-" + str(self.__log_reg.C) + "-" + str(self.__log_reg.max_iter)
                content = "*----VALIDATION----*\nSolver: " + self.__log_reg.solver + "\nregularization: " + str(self.__log_reg.C) + "\niterations: " + str(self.__log_reg.max_iter) + "\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
                self.__output_results(filename, content)

                if f1 > f1_best:
                    f1_best = f1
                    accuracy_best = accuracy
                    solver_best = solver
                    reg_best = reg

        print("*----BEST----*")
        print("f1:", f1_best)
        print("accuracy:", accuracy_best) 
        print("solver:", solver_best)
        print("regularization:", reg_best)

        # Save output to file
        filename = "best-" + solver_best + "-" + str(reg_best) + "-" + str(self.__log_reg.max_iter)
        content = "*----BEST----*\nSolver: " + solver_best + "\nregularization: " + str(reg_best) + "\niterations: " + str(self.__log_reg.max_iter) + "\nF1: " + str(f1_best) + "\naccuracy: " + str(accuracy_best)
        self.__output_results(filename, content)

        # Loading previously trained model
        # self.__load_model("finalized_model_test.sav")
        
        # Testing
        self.__train_logistic_regression_model(solver_best, reg_best, nIterations)
        self.__test_logistic_regression()

        
        seconds_passed = time.time() - self.__start_time
        print("[INFO] Program finished after", str(int(seconds_passed / 60)) + ":" + str(seconds_passed % 60), "minutes")

    def __train_logistic_regression_model(self, new_solver = "lbfgs", regularization = 1.0, nIterations = 1000):
        self.__log_reg.solver = new_solver
        self.__log_reg.C = regularization
        self.__log_reg.max_iter = nIterations

        print("\n*--------------------TRAINING--------------------*")
        print("[INFO] Learning from", self.__training_set_positive_examples_count + self.__training_set_negative_examples_count, "images in total.")

        # Shuffling data
        A = np.c_[ self.__X_training, self.__y_training]
        np.random.shuffle(A)
        num_cols = np.shape(A)[1]

        self.__X_training = A[:, 0:num_cols - 1]
        self.__y_training = A[:, num_cols - 1]

        # Training
        print("[INFO] Training started.")
        self.__log_reg.fit(self.__X_training, self.__y_training)
        print("[INFO] Training done.")
        
        self.__save_model()

    def __load_model(self, path):
        pickle.load(open(path, "rb"))
        
    def __validate_logistic_regression(self):
        print("\n*--------------------VALIDATING--------------------*")
        print("[INFO] Validating on", self.__validation_set_positive_examples_count + self.__validation_set_negative_examples_count, "images in total.")

        # Shuffling data
        A = np.c_[ self.__X_cv, self.__y_cv]
        np.random.shuffle(A)
        num_cols = np.shape(A)[1]

        self.__X_cv = A[:, 0:num_cols - 1]
        self.__y_cv = A[:, num_cols - 1]

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

        # Shuffling data
        A = np.c_[ self.__X_test, self.__y_test]
        np.random.shuffle(A)
        num_cols = np.shape(A)[1]

        self.__X_test = A[:, 0:num_cols - 1]
        self.__y_test = A[:, num_cols - 1]

        # Testing
        print("[INFO] Testing started.")
        accuracy = self.__log_reg.score(self.__X_test, self.__y_test)
        y_predicted = np.zeros(self.__y_test.shape, dtype = float)
        i = 0
        for example in self.__X_test:
            example = example[np.newaxis, :]
            y_predicted[i] = self.__log_reg.predict(example)
            i += 1
        f1 = f1_score(self.__y_test, y_predicted) 
        print("[INFO] Testing done.")

        print("\n*--------------------RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy * 100, "%")
        print("[INFO] F1 score obtained on the test set:", f1)

    def __output_results(self, filename, content):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        file_path = self.__output_dir + "/" + filename + "_" + str(current_time) + ".txt"
        with open(file_path, 'w+') as f:
            f.write(content)