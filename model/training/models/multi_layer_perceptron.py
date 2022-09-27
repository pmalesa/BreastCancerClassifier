import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from sklearn.metrics import f1_score, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc, confusion_matrix

from image_processing.image_manager import ImageManager
from io_module.io_manager import IOManager

from datetime import datetime

import matplotlib.pyplot as plt

import statistics
import numpy as np
import pickle
import os
import time

class MultiLayerPerceptron:
    def __init__(self, activation = "logistic",  verbose = True):
        self.__model = keras.models.Sequential()
        self.__epochs = 1000
        self.__callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", mode = "min", patience = 3) # 5 is too much

        self.__dirty = False

        self.__im = ImageManager()
        self.__io = IOManager()

        # self.__positive_examples_count = im.get_positive_examples_count()
        # self.__negative_examples_count = im.get_negative_examples_count()

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
        self.__finalized_model_filename = "mlp_finalized_model.sav"

        self.__train_loss_avg = np.array([])
        self.__test_loss = np.array([])

        self.__train_acc = np.array([])
        self.__test_acc = np.array([])

        self.__avg_classification_time = 0.0
        self.__std_classification_time = 0.0

    def save_model(self, directory):
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        self.__model.save(directory + "/" + self.__finalized_model_filename + "_" + str(current_time) + ".sav")
        
    def load_model(self, path):
        self.__model  = keras.models.load_model(path)
        
    def start_learning_process(self):
        log_message = "*-----(MLP_LEARNING_PROCESS_STARTED)-----*"
        self.__io.append_log(log_message)

        # learning_rates = [0.00001, 0.0001, 0.001] #, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        # best_accuracy = 0.0
        # best_learning_rate = 0.0
        # for learning_rate in learning_rates:
        #     self.__train(dataset = "train", learning_rate = learning_rate, epochs = 1000)
        #     (accuracy, f1) = self.__validate()
        #     log_message = "(Validation) learning_rate = " + str(learning_rate) + "; accuracy = " + str(accuracy) + "; f1 = " + str(f1) + ";"
        #     self.__io.append_log(log_message)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_learning_rate = learning_rate

        best_learning_rate = 0.00001

        start = datetime.now()
        self.__train(dataset = "full", learning_rate = best_learning_rate, epochs = 1000)
        finish = datetime.now()
        duration = finish - start
        training_time_in_s = duration.total_seconds() # [seconds]

        (test_accuracy, test_f1, confusion_matrix, classif_time_stats) = self.__test()

        tn, fp, fn, tp = confusion_matrix
        mean_classif_time, std_classif_time = classif_time_stats

        # Saving results
        log_message = f"(Testing) [learning_rate = {best_learning_rate}]:\n \
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
        filename = f"mlp_metrics_results_{current_time}_{best_learning_rate}.txt"
        content = log_message
        self.__io.save_results("mlp", filename, content)

    def __initialize_mlp_model(self, learning_rate = 0.0001):
        del self.__model
        self.__model = keras.models.Sequential()
        self.__model.add(keras.layers.Dense(10000, input_shape = [7500], activation = "sigmoid"))
        self.__model.add(keras.layers.Dense(10000, activation = "sigmoid"))
        self.__model.add(keras.layers.Dense(1, activation = "sigmoid"))
        self.__model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"])
        self.__dirty = False

    def __train(self, dataset = "train", learning_rate = 0.0001, epochs = 1000):
        print("\n*--------------------TRAINING-MLP--------------------*")
        print("[INFO] Training of multi-layer perceptron model started.")

        self.__initialize_mlp_model(learning_rate)

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
            history = self.__model.fit(X_training, y_training, epochs = self.__epochs, shuffle = True, callbacks = [self.__callback])

            if dataset == "full":
                # Train loss
                self.__train_loss_avg = np.append(self.__train_loss_avg, statistics.mean(history.history["loss"]))

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("[INFO] Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")

        self.__dirty = True

        print("[INFO] Training done.")
        print("[INFO] Total images processed:", positive_images_processed + negative_images_processed)

    def __validate(self):
        print("\n*--------------------VALIDATING-MLP--------------------*")
        print("[INFO] Validation of multi-layer perceptron model started.")
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
        accuracy = self.__model.evaluate(X_cv, y_cv)
        y_predicted_prob = self.__model.predict(X_cv)
        y_predicted = self.__classify_predictions(y_predicted_prob)
        f1 = self.__calculate_f1(y_predicted, y_cv)
        print(f"[INFO] Validation of MLP model finished.")

        return (accuracy[1], f1)

    def __test(self):
        print("\n*--------------------TESTING-MLP--------------------*")
        print("[INFO] Testing of multi-layer perceptron model started.")
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
        accuracy = self.__model.evaluate(X_test, y_test)
        y_predicted_prob = self.__model.predict(X_test)
        y_predicted = self.__classify_predictions(y_predicted_prob)
        f1 = self.__calculate_f1(y_predicted, y_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

        # Time benchmark
        test_examples_count = self.__test_set_positive_examples_count + self.__test_set_negative_examples_count
        mean_classif_time = 0.0 # [s]
        std_classif_time = 0.0 # [s]
        classif_times = np.array([])

        count = 0
        for i in range(test_examples_count):
            test_example = X_test[i][np.newaxis, ...]
            start = datetime.now()
            prediction = self.__model.predict(test_example)
            finish = datetime.now()
            duration = finish - start
            classif_times = np.append(classif_times, duration.total_seconds())
            count += 1
            if count > 1000:
                break

        mean_classif_time = np.mean(classif_times)
        std_classif_time = np.std(classif_times)   

        print("[INFO] Testing done.")
        print("[INFO] Tested on ", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "examples --->", self.__test_set_positive_examples_count, "positive and", self.__test_set_negative_examples_count, "negative.")
        print("\n*--------------------MLP-TESTING-RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy[1])
        print("[INFO] F1 score obtained on the test set:", f1)

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize = (10, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, y_predicted, display_labels = ["0", "1"], ax = ax)
        plt.show()
        fig.savefig('mlp_confusion_matrix.png')

        # ROC curve
        fig, ax = plt.subplots(figsize = (10, 8))
        fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='MLP (AUC = {:.2f})'.format(roc_auc), color = "b")
        plt.xlabel('False Positive Rate (Positive label: 1.0)')
        plt.ylabel('True Positive Rate (Positive label: 1.0)')
        plt.title('ROC curve')
        plt.legend(loc='lower right')
        plt.show()
        fig.savefig('mlp_roc.png')

        # Train loss
        x = np.linspace(1000, 1000 + self.__train_loss_avg.size * 1000, self.__train_loss_avg.size, endpoint = False)
        fig, ax = plt.subplots(figsize = (10, 8))
        plt.plot(x, self.__train_loss_avg, color = "b")
        plt.title("MLP model loss function value during training")
        plt.ylabel("Loss")
        plt.xlabel("Training examples")
        plt.legend(["Train loss"], loc = "upper right")
        plt.show()
        fig.savefig('mlp_loss.png')

        # Train and test accuracies
        x = np.linspace(1000, 1000 + self.__train_acc.size * 1000, self.__train_acc.size, endpoint = False)
        fig, ax = plt.subplots(figsize = (10, 8))
        plt.plot(x, self.__train_acc, color = "b")
        plt.plot(x, self.__test_acc, color = "r")
        plt.title("MLP model train and test accuracies during training")
        plt.ylabel("Accuracy")
        plt.xlabel("Training examples")
        plt.legend(["Train accuracy", "Test accuracy"], loc = "upper right")
        plt.show()
        fig.savefig('mlp_accuracy.png')

        return (accuracy[1], f1, [tn, fp, fn, tp], [mean_classif_time, std_classif_time])

    def __classify_predictions(self, y_pred_prob):
        y_pred_list = list()
        for i in range(0, len(y_pred_prob)):
            if y_pred_prob[i] >= 0.5:
                y_pred_list.append(1.0)
            else:
                y_pred_list.append(0.0)
        y_pred = np.asarray(y_pred_list)
        return y_pred

    def __calculate_f1(self, y_pred, y_real):
        '''
        Calculates f1 score using two given vectors.
        Returns -1 if f1 can not be calculated.
        '''
        if len(y_pred) != len(y_real):
            return -1

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, len(y_pred)):
            if y_pred[i] == 1 and y_real[i] == 1:
                true_positives += 1
            elif y_pred[i] == 0 and y_real[i] == 1:
                false_negatives += 1
            elif y_pred[i] == 1 and y_real[i] == 0:
                false_positives += 1

        if true_positives + false_positives == 0:
            return -1
        else:
            precision = true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives == 0:
            return -1
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            return -1
        else:
            return 2 * precision * recall / (precision + recall)