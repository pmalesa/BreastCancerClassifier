import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from image_processing.image_manager import ImageManager
from io_module.io_manager import IOManager

from datetime import datetime

import numpy as np
import os
import time

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation = "relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides = strides, padding = "same", use_bias = False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides = 1, padding = "same", use_bias = False),
            keras.layers.BatchNormalization()] 
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides = strides, padding = "same", use_bias = False),
                keras.layers.BatchNormalization()]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)    

class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.__model = keras.models.Sequential()

        self.__dirty = False

        self.__im = ImageManager()
        self.__io = IOManager()


        # self.__positive_examples_count = self.__im.get_positive_examples_count()
        # self.__negative_examples_count = self.__im.get_negative_examples_count()

        # self.__positive_examples_count = 78769
        # self.__negative_examples_count = 78769

        # Cleaned data -> 78766 example count

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
        self.__finalized_model_filename = "cnn_finalized_model.sav"

    def save_model(self, directory):
        self.__model.save(directory + "/" + self.__finalized_model_filename)

    def load_model(self, path):
        self.__model = keras.models.load_model(path)

    def start_learning_process(self):
        # self.__train(learning_rate = 0.01, epochs = 10)
        # (test_accuracy, test_f1) = self.__test()

        # log_message = f"(Testing on cleaned data) learning_rate = 0.01; epochs = 10; accuracy = {test_accuracy}; f1 = {test_f1}"
        # self.__io.append_log(log_message)

        log_message = "*-----(CNN_LEARNING_PROCESS_STARTED)-----*"
        self.__io.append_log(log_message)
        learning_rates = [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0 ]
        epochs_values = [ 10, 20, 50, 100]

        best_accuracy = 0.0
        best_learning_rate = 0.0
        best_epochs_value = 0

        for learning_rate in learning_rates:
            for epochs in epochs_values:
                self.__train(learning_rate = learning_rate, epochs = epochs)
                (accuracy, f1) = self.__validate()
                log_message = f"(Validation) learning_rate = {learning_rate}; epochs = {epochs}; accuracy = {accuracy}; f1 = {f1}" 
                self.__io.append_log(log_message)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_learning_rate = learning_rate
                    best_epochs_value = epochs

        print(f"[INFO] BEST TRAINING PARAMETERS:\nlearning_rate = {best_learning_rate}; epochs = {epochs}")
        self.__train(learning_rate = best_learning_rate, epochs = best_epochs_value)
        (test_accuracy, test_f1) = self.__test()
        log_message = f"(Testing) learning_rate = {best_learning_rate}; epochs = {best_epochs_value}; accuracy = {test_accuracy}; f1 = {test_f1}"
        self.__io.append_log(log_message)

    def __initialize_resnet_model(self, learning_rate):
        if self.__dirty == True:
            del self.__model
            self.__model = keras.models.Sequential()
            
        self.__model.add(keras.layers.Conv2D(64, 7, strides = 2, input_shape = [50, 50, 3], padding = "same", use_bias = False))
        self.__model.add(keras.layers.BatchNormalization())
        self.__model.add(keras.layers.Activation("relu"))
        self.__model.add(keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same"))
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.__model.add(ResidualUnit(filters, strides = strides))
            prev_filters = filters
        self.__model.add(keras.layers.GlobalAvgPool2D())
        self.__model.add(keras.layers.Flatten())
        self.__model.add(keras.layers.Dense(1, activation = "sigmoid")) # must be sigmoid
        self.__model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"])        

    def __train(self, learning_rate = 0.01, epochs = 10):
        print("\n*--------------------TRAINING-CONVOLUTIONAL-NEURAL-NETWORK--------------------*")
        print("[INFO] Training of convolutional neural network model (ResNet-34) started.")
        print("[INFO] Learning from", self.__training_set_positive_examples_count + self.__training_set_negative_examples_count, "images in total.")

        gpu_devices = tf.config.list_physical_devices('GPU')
        print("[INFO] Number of GPUs available:", len(gpu_devices))

        self.__initialize_resnet_model(learning_rate)

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

            X_training = np.concatenate((X_pos, X_neg), axis = 0)
            y_training = np.hstack([y_pos, y_neg])

            # Learning from batch
            history = self.__model.fit(X_training, y_training, epochs = epochs, shuffle = True)

            self.__dirty = True

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("[INFO] Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")

        print("[INFO] Training done.")
        print("[INFO] Total images processed:", positive_images_processed + negative_images_processed)

    def __validate(self):
        print("\n*--------------------VALIDATING-CONVOLUTIONAL-NEURAL-NETWORK--------------------*")
        print("[INFO] Validation of convolutional neural network model (ResNet-34) started.")
        print("[INFO] Validating on", self.__validation_set_positive_examples_count + self.__validation_set_negative_examples_count, "images in total.")

        first_positive_image = self.__training_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__validation_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__validation_set_negative_examples_count

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data(first_negative_image, last_negative_image, 0)

        X_cv = np.concatenate((X_pos, X_neg), axis = 0)
        y_cv = np.hstack([y_pos, y_neg])

        # Validating the model by computing accuracy and f1 metrics
        accuracy = self.__model.evaluate(X_cv, y_cv)
        y_predicted_prob = self.__model.predict(X_cv)
        y_predicted = self.__classify_predictions(y_predicted_prob)
        f1 = self.__calculate_f1(y_predicted, y_cv)
        print("[INFO] Validation of convolutional neural network model (ResNet-34) finished.")

        return (accuracy[1], f1)

    def __test(self):
        print("\n*--------------------TESTING-CONVOLUTIONAL-NEURAL-NETWORK--------------------*")
        print("[INFO] Testing of convolutional neural network model (ResNet-34) started.")
        print("[INFO] Testing on", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "images in total.")

        first_positive_image = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count + 1 
        last_positive_image = first_positive_image + self.__test_set_positive_examples_count
        first_negative_image = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count + 1 
        last_negative_image = first_negative_image + self.__test_set_negative_examples_count

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data(first_positive_image, last_positive_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data(first_negative_image, last_negative_image, 0)

        X_test = np.concatenate((X_pos, X_neg), axis = 0)
        y_test = np.hstack([y_pos, y_neg])

        # Testing on a batch
        accuracy = self.__model.evaluate(X_test, y_test)
        y_predicted_prob = self.__model.predict(X_test)
        y_predicted = self.__classify_predictions(y_predicted_prob)
        f1 = self.__calculate_f1(y_predicted, y_test)

        print("[INFO] Testing done.")
        print("[INFO] Tested on ", self.__test_set_positive_examples_count + self.__test_set_negative_examples_count, "examples --->", self.__test_set_positive_examples_count, "positive and", self.__test_set_negative_examples_count, "negative.")

        print("\n*--------------------CONVOLUTIONAL-NEURAL-NETWORK-TESTING-RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy[1])
        print("[INFO] Scalar test loss:", accuracy[0])
        print("[INFO] F1 score obtained on the test set:", f1)

        # Save output to file
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        filename = "resnet_cnn_metrics_results_" + str(current_time) + ".txt"
        content = "*----RESULTS----*\n\nF1: " + str(f1) + "\naccuracy: " + str(accuracy)
        self.__io.save_results("cnn", filename, content)

        return (accuracy[1], f1)

        # -----------------------------------------------------------------
        # file_path = "./output/logistic_regression_classification_results" + current_time + ".txt"
        # f = open(file_path, 'w+')
        # f.write("PRED     REAL\n")
        # for i in range(0, y_test.shape[0]):
        #    f.write(str(y_predicted[i]) + "    " + str(y_test[i]) + "\n")
        # f.close()
        # -----------------------------------------------------------------

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


