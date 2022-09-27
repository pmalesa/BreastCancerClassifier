from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, RocCurveDisplay, log_loss, confusion_matrix

from image_processing.image_manager import ImageManager
from io_module.io_manager import IOManager

from datetime import datetime

import matplotlib.pyplot as plt

import statistics
import numpy as np
import pickle
import os
import time

class LogististicRegression:
    def __init__(self, loss = "log", shuffle = True, verbose = 1, eta0 = 1.0, warm_start = True):
        self.__log_reg = SGDClassifier(loss = loss, shuffle = shuffle, verbose = verbose, eta0 = eta0, warm_start = warm_start)

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

        self.__positive_examples_count = 78769  # max = 78769
        self.__negative_examples_count = 78769  # max = 78769

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

        self.__train_loss = np.array([])
        self.__test_loss = np.array([])

        self.__train_acc = np.array([])
        self.__test_acc = np.array([])

        self.__avg_classification_time = 0.0
        self.__std_classification_time = 0.0
    
    def save_model(self, directory):
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        pickle.dump(self.__log_reg, open(directory + "/" + current_time + "_" + self.__finalized_model_filename, "wb"))

    def load_model(self, path):
        self.__log_reg = pickle.load(open(path, "rb"))

    def start_learning_process(self):
        log_message = "*-----(LOGISTIC_REGRESSION_LEARNING_PROCESS_STARTED)-----*"
        self.__io.append_log(log_message)
        # alpha_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        # best_accuracy = 0.0
        # best_alpha = 0.0
        # for alpha in alpha_values:
        #     self.__train(dataset = "train", alpha = alpha)
        #     (accuracy, f1) = self.__validate()
        #     log_message = "(Validation) alpha = " + str(alpha) + " : accuracy = " + str(accuracy) + "; f1 = " + str(f1) + ";"
        #     self.__io.append_log(log_message)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_alpha = alpha

        # print("[INFO] BEST TRAINING PARAMETERS: \n   -alpha: " + str(best_alpha))
        # self.__train(alpha = best_alpha)

        best_alpha = 0.001

        start = datetime.now()
        self.__train(dataset = "full", alpha = best_alpha)
        finish = datetime.now()
        duration = finish - start
        training_time_in_s = duration.total_seconds() # [seconds]
        
        (test_accuracy, test_f1, confusion_matrix, classif_time_stats) = self.__test()

        tn, fp, fn, tp = confusion_matrix
        mean_classif_time, std_classif_time = classif_time_stats

        # Saving results
        log_message = f"(Testing) [alpha = {best_alpha}]:\n \
                - accuracy:  {test_accuracy}\n \
                - f1: {test_f1}\n \
                - TP: {tp}\n \
                - TN: {tn}\n \
                - FP: {fp}\n \
                - FN: {fn}\n \
                - TNR (Specificity): {tn / (tn + fp)}\n \
                - TPR (Recall/Sensitivity): {tp / (tp + fn)}\n \
                - FPR (Type I error): {fp / (fp + tn)}\n \
                - FNR (Type II error): {fn / (tp + fn)}\n \
                - PPV (Precision): {tp / (tp + fp)}\n \
                - NPV (Neg. Pred. Val.): {tn / (tn + fn)}\n \
                - Average classification time [ms]: {mean_classif_time * 1000}\n \
                - Standard deviation of classification time [ms]: {std_classif_time * 1000}\n \
                - Training time [min]: {training_time_in_s / 60}\n"
        self.__io.append_log(log_message)
        print(log_message)

        # Save output to file
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        filename = "logistic_regression_metrics_results_" + str(current_time) + "_" + str(self.__alpha) + ".txt"
        content = log_message
        self.__io.save_results("logistic_regression", filename, content)

    def __reset(self):
        del self.__log_reg
        self.__log_reg = SGDClassifier(loss = self.__loss, shuffle = self.__shuffle, verbose = self.__verbose, eta0 = self.__eta0, warm_start = self.__warm_start)

    def __train(self, dataset = "train", learning_rate = "optimal", alpha = 0.0001, iterations = 1000):
        print("\n*--------------------TRAINING-LOGISTIC-REGRESSION--------------------*")
        print("[INFO] Training of logistic regression model started.")

        # Reinitialize SGDClassifier object and initialize hyperparameters
        if self.__dirty == True:
            self.__reset()
        self.__log_reg.learning_rate = learning_rate
        self.__alpha = alpha
        self.__log_reg.alpha = alpha
        self.__log_reg.max_iter = iterations

        # Loading 10000 image examples for testing
        first_positive_test_image = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count + 1 
        last_positive_test_image = first_positive_test_image + 5000
        first_negative_test_image = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count + 1 
        last_negative_test_image = first_negative_test_image + 5000

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data_flattened(first_positive_test_image, last_positive_test_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data_flattened(first_negative_test_image, last_negative_test_image, 0)

        X_test = np.vstack([X_pos, X_neg])
        y_test = np.hstack([y_pos, y_neg])

        # Preparing the training data
        positive_images_processed = 0
        negative_images_processed = 0

        if dataset == "train":
            total_positive_image_count = self.__training_set_positive_examples_count
            total_negative_image_count = self.__training_set_negative_examples_count
        elif dataset == "full":     # train set + cross-validation set
            total_positive_image_count = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count
            total_negative_image_count = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count

        print("[INFO] Learning from", total_positive_image_count + total_negative_image_count, "images in total.")

        while positive_images_processed < total_positive_image_count or negative_images_processed < total_negative_image_count:
            positive_images_left = total_positive_image_count - positive_images_processed
            negative_images_left = total_negative_image_count - negative_images_processed

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

            if dataset == "full":
                # Train and test losses
                y_predicted = self.__log_reg.predict(X_training)
                self.__train_loss = np.append(self.__train_loss, log_loss(y_training, y_predicted))
                y_predicted = self.__log_reg.predict(X_test)
                self.__test_loss = np.append(self.__test_loss, log_loss(y_test, y_predicted))

                # Train and test accuracies
                train_acc = self.__log_reg.score(X_training, y_training)
                test_acc = self.__log_reg.score(X_test, y_test)
                self.__train_acc = np.append(self.__train_acc, train_acc)
                self.__test_acc = np.append(self.__test_acc, test_acc)     

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("[INFO] Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")
        
        self.__dirty = True
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

        # Loading image data for testing
        first_positive_image = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__test_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__test_set_negative_examples_count

        (X_pos, y_pos) = self.__im.load_image_data_flattened(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data_flattened(first_negative_image, last_negative_image, 0)

        X_test = np.vstack([X_pos, X_neg])
        y_test = np.hstack([y_pos, y_neg])

        # Testing on a batch
        accuracy = self.__log_reg.score(X_test, y_test)
        y_predicted = self.__log_reg.predict(X_test)
        f1 = f1_score(y_test, y_predicted)

        tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

        # Time benchmark
        test_examples_count = self.__test_set_positive_examples_count + self.__test_set_negative_examples_count
        mean_classif_time = 0.0 # [s]
        std_classif_time = 0.0 # [s]
        classif_times = np.array([])

        for i in range(test_examples_count):
            test_example = X_test[i].reshape(1, -1)
            start = datetime.now()
            prediction = self.__log_reg.predict(test_example)
            finish = datetime.now()
            duration = finish - start
            classif_times = np.append(classif_times, duration.total_seconds())

        mean_classif_time = np.mean(classif_times)
        std_classif_time = np.std(classif_times)        

        print("[INFO] Testing done.")
        print("[INFO] Tested on ", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "examples --->", self.__test_set_positive_examples_count, "positive and", self.__test_set_negative_examples_count, "negative.")
        print("\n*--------------------LOGISTIC-REGRESSION-TESTING-RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy)
        print("[INFO] F1 score obtained on the test set:", f1)

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize = (10, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, y_predicted, display_labels = ["0", "1"], ax = ax)
        plt.show()

        # ROC curve
        fig, ax = plt.subplots(figsize = (10, 8))
        RocCurveDisplay.from_estimator(self.__log_reg, X_test, y_test, color = "b", ax = ax)
        plt.show()

        # Train and test loss
        x = np.linspace(1000, 1000 + self.__train_loss.size * 1000, self.__train_loss.size, endpoint = False)
        fig, ax = plt.subplots(figsize = (10, 8))
        plt.plot(x, self.__train_loss, color = "b")
        plt.plot(x, self.__test_loss, color = "r")
        plt.title("Logistic regression model train and test loss during training")
        plt.ylabel("Loss")
        plt.xlabel("Training examples")
        plt.legend(["Train loss", "Test loss"], loc = "upper right")
        plt.show()

        # Train and test accuracies
        x = np.linspace(1000, 1000 + self.__train_acc.size * 1000, self.__train_acc.size, endpoint = False)
        fig, ax = plt.subplots(figsize = (10, 8))
        plt.plot(x, self.__train_acc, color = "b")
        plt.plot(x, self.__test_acc, color = "r")
        plt.title("Logistic regression model train and test accuracies during training")
        plt.ylabel("Accuracy")
        plt.xlabel("Training examples")
        plt.legend(["Train accuracy", "Test accuracy"], loc = "upper right")
        plt.show()

        return (accuracy, f1, [tn, fp, fn, tp], [mean_classif_time, std_classif_time])