from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from image_processing.image_manager import ImageManager
from io_module.io_manager import IOManager

from datetime import datetime

import numpy as np
import pickle
import os
import time

class SupportVectorMachine:
    def __init__(self, loss = "hinge", shuffle = True, verbose = 1, eta0 = 1.0, warm_start = True):
        self.__svm = SGDClassifier(loss = loss, shuffle = shuffle, verbose = verbose, eta0 = eta0, warm_start = warm_start)

        # Model parameters
        self.__loss = loss
        self.__shuffle = shuffle
        self.__verbose = verbose
        self.__eta0 = eta0
        self.__warm_start = warm_start
        self.__alpha = 0.0001

        self.__dirty = False

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
        self.__finalized_model_filename = "svm_finalized_model.sav"

    def save_model(self, directory):
        pickle.dump(self.__svm, open(directory + "/" + self.__finalized_model_filename, "wb"))

    def load_model(self, path):
        self.__svm = pickle.load(open(path, "rb"))

    def start_learning_process(self):
        log_message = "*-----(SVM_LEARNING_PROCESS_STARTED)-----*"
        self.__io.append_log(log_message)
        alpha_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        best_accuracy = 0.0
        best_alpha = 0.0
        for alpha in alpha_values:
            self.__train(alpha = alpha)
            (accuracy, f1) = self.__validate()
            log_message = "(Validation) alpha = " + str(alpha) + "; accuracy = " + str(accuracy) + "; f1 = " + str(f1) + ";"
            self.__io.append_log(log_message)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha

        print("[INFO] BEST TRAINING PARAMETERS: \n   -alpha: " + str(best_alpha))
        self.__train(alpha = best_alpha)
        (test_accuracy, test_f1) = self.__test()
        log_message = "(Testing) best_alpha = " + str(best_alpha) + " : accuracy = " + str(test_accuracy) + "; f1 = " + str(test_f1) + ";"
        self.__io.append_log(log_message)

    def __reset(self):
        del self.__svm
        self.__svm = SGDClassifier(loss = self.__loss, shuffle = self.__shuffle, verbose = self.__verbose, eta0 = self.__eta0, warm_start = self.__warm_start)

    def __train(self, learning_rate = "optimal", alpha = 0.0001, iterations = 1000):
        print("\n*--------------------TRAINING-SVM--------------------*")
        print("[INFO] Training of SVM model started.")
        print("[INFO] Learning from", self.__training_set_positive_examples_count + self.__training_set_negative_examples_count, "images in total.")

        # Reinitialize SGDClassifier object and initialize hyperparameters
        if self.__dirty == True:
            self.__reset()
        self.__svm.learning_rate = learning_rate
        self.__alpha = alpha
        self.__svm.alpha = alpha
        self.__svm.max_iter = iterations

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
            self.__svm.fit(X_training, y_training)

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("[INFO] Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")

        self.__dirty = True
        print("[INFO] Training done.")
        print("[INFO] Total images processed:", positive_images_processed + negative_images_processed)

    def __validate(self):
        print("\n*--------------------VALIDATING-SVM--------------------*")
        print("[INFO] Validation of SVM model started.")
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
        accuracy = self.__svm.score(X_cv, y_cv)
        y_predicted = self.__svm.predict(X_cv)
        f1 = f1_score(y_cv, y_predicted)
        print("[INFO] Validation of SVM model finished.")

        return (accuracy, f1)

    def __test(self):
        print("\n*--------------------TESTING-SVM--------------------*")
        print("[INFO] Testing of SVM model started.")
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
        accuracy = self.__svm.score(X_test, y_test)
        y_predicted = self.__svm.predict(X_test)
        f1 = f1_score(y_test, y_predicted)

        print("[INFO] Testing done.")
        print("[INFO] Tested on ", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "examples --->", self.__test_set_positive_examples_count, "positive and", self.__test_set_negative_examples_count, "negative.")

        print("\n*--------------------SVM-TESTING-RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy)
        print("[INFO] F1 score obtained on the test set:", f1)

        # Save output to file
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        filename = "svm_metrics_results_" + str(current_time) + "_" + str(self.__alpha) + ".txt"
        content = "*----RESULTS----*\n\nC = " + str(self.__alpha) + "\n\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        self.__io.save_results("svm", filename, content)

        return (accuracy, f1)