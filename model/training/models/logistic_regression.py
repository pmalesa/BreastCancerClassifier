from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from image_processing.image_manager import ImageManager
from io_module.io_manager import IOManager

from datetime import datetime

import numpy as np
import pickle
import os
import time

class LogististicRegression:
    def __init__(self, loss = "log", shuffle = True, verbose = 1, eta0 = 1.0, warm_start = True):
        self.__log_reg = SGDClassifier(loss = loss, shuffle = shuffle, verbose = verbose, eta0 = eta0, warm_start = warm_start)

        self.__im = ImageManager()
        self.__io = IOManager()

        # self.__positive_examples_count = im.get_positive_examples_count()
        # self.__negative_examples_count = im.get_negative_examples_count()

        # self.__positive_examples_count = 78769
        # self.__negative_examples_count = 78769

        self.__positive_examples_count = 78769
        self.__negative_examples_count = 78769

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
        self.__finalized_model_filename = "logistic_regression_finalized_model.sav"
    
    def save_model(self, directory):
        pickle.dump(self.__log_reg, open(directory + "/" + self.__finalized_model_filename, "wb"))

    def load_model(self, path):
        self.__log_reg = pickle.load(open(path, "rb"))

    def start_learning_process(self):
        self.__train()
        self.__test()
        # TBC - here I will test and validate the model for different hyperparameters

    def __reset(self):
        self.__log_reg = SGDClassifier(loss = self.__log_reg.loss, shuffle = self.__log_reg.shuffle,
            verbose = self.__log_reg.verbose, eta0 = self.__log_reg.eta0, warm_start = self.__log_reg.warm_start)

    def __train(self, learning_rate = "optimal", regularization = 0.0001, iterations = 1000):
        print("\n*--------------------TRAINING-LOGISTIC-REGRESSION--------------------*")
        print("[INFO] Training of logistic regression model started.")
        print("[INFO] Learning from", self.__training_set_positive_examples_count + self.__training_set_negative_examples_count, "images in total.")

        # Reinitialize SGDClassifier object and initialize hyperparameters
        self.__reset()
        self.__log_reg.learning_rate = learning_rate
        self.__log_reg.alpha = regularization
        self.__log_reg.max_iter = iterations

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
            (X_pos, y_pos) = self.__im.load_image_data_flattened(first_positive_image, last_positive_image, 1)
            (X_neg, y_neg) = self.__im.load_image_data_flattened(first_negative_image, last_negative_image, 0)

            X_training = np.vstack([X_pos, X_neg])
            y_training = np.hstack([y_pos, y_neg])

            # Shuffling data
            A = np.c_[X_training, y_training]
            np.random.shuffle(A)
            num_cols = np.shape(A)[1]

            X_training = A[:, 0:num_cols - 1]
            y_training = A[:, num_cols - 1]

            # Learning from batch
            self.__log_reg.fit(X_training, y_training)

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("[INFO] Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")

        print("[INFO] Training done.")
        print("[INFO] Total images processed:", positive_images_processed + negative_images_processed)

    def __validate(self):
        print("\n*--------------------VALIDATING-LOGISTIC-REGRESSION--------------------*")
        print("[INFO] Validation of logistic regression model started.")
        print("[INFO] Validating on", self.__validation_set_positive_examples_count + self.__validation_set_negative_examples_count, "images in total.")

        first_positive_image = self.__training_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__validation_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__validation_set_negative_examples_count

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data_flattened(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data_flattened(first_negative_image, last_negative_image, 0)

        X_cv = np.vstack([X_pos, X_neg])
        y_cv = np.hstack([y_pos, y_neg])

        # Validating the model by computing accuracy and f1 metrics
        accuracy = self.__log_reg.score(X_cv, y_cv)
        y_predicted = self.__log_reg.predict(X_cv)
        f1 = f1_score(y_cv, y_predicted)
        print("[INFO] Validation of logistic regression model finished.")

        return (accuracy, f1)

    def __test(self):
        print("\n*--------------------TESTING-LOGISTIC-REGRESSION--------------------*")
        print("[INFO] Testing of logistic regression model started.")
        print("[INFO] Testing on", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "images in total.")

        first_positive_image = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__test_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__test_set_negative_examples_count

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data_flattened(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data_flattened(first_negative_image, last_negative_image, 0)

        X_test = np.vstack([X_pos, X_neg])
        y_test = np.hstack([y_pos, y_neg])

        # Testing on a batch
        accuracy = self.__log_reg.score(X_test, y_test)
        y_predicted = self.__log_reg.predict(X_test)
        f1 = f1_score(y_test, y_predicted)

        print("[INFO] Testing done.")
        print("[INFO] Tested on ", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "examples --->", self.__test_set_positive_examples_count, "positive and", self.__test_set_negative_examples_count, "negative.")

        print("\n*--------------------LOGISTIC-REGRESSION-TESTING-RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy)
        print("[INFO] F1 score obtained on the test set:", f1)

        # Save output to file
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        filename = "logistic_regression_metrics_results_" + str(current_time) + ".txt"
        content = "*----RESULTS----*\n\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        self.__io.save_results(filename, content)

        # -----------------------------------------------------------------
        # file_path = "./output/logistic_regression_classification_results" + current_time + ".txt"
        # f = open(file_path, 'w+')
        # f.write("PRED     REAL\n")
        # for i in range(0, y_test.shape[0]):
        #    f.write(str(y_predicted[i]) + "    " + str(y_test[i]) + "\n")
        # f.close()
        # -----------------------------------------------------------------