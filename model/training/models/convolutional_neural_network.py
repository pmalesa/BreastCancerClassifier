import sys

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Add, Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPool2D #

from sklearn.metrics import f1_score, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc, confusion_matrix

from image_processing.image_manager import ImageManager
from io_module.io_manager import IOManager

from datetime import datetime

import matplotlib.pyplot as plt

import statistics
import numpy as np
import os
import time

def identity_block(X, f, filters):
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(X)
    X = BatchNormalization()(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X   

def convolutional_block(X, f, filters, s):
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s, s), padding = 'valid')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides=(1, 1), padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides=(1, 1), padding = 'valid')(X)
    X = BatchNormalization()(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s, s), padding = 'valid')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(learning_rate = 0.01, input_shape = (50, 50, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(1, activation = "sigmoid")(X)
    
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"])        

    return model

def ResNet101(learning_rate = 0.01, input_shape = (50, 50, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    for i in range(3):
        X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    for i in range(22):
        X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(1, activation = "sigmoid")(X)
    
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"])        

    return model

def ResNet152(learning_rate = 0.01, input_shape = (50, 50, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    for i in range(7):
        X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    for i in range(35):
        X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(1, activation = "sigmoid")(X)
    
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"])        

    return model

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
        self.__variant = "resnet-34"

        self.__dirty = False

        self.__im = ImageManager()
        self.__io = IOManager()

        # This callback will stop the training when there is no improvement in the loss for three consecutive epochs
        self.__callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", mode = "min", patience = 3) # 5 is too much

        # self.__positive_examples_count = self.__im.get_positive_examples_count()
        # self.__negative_examples_count = self.__im.get_negative_examples_count()

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
        self.__finalized_model_filename = "cnn_finalized_model"

        self.__train_loss_avg = np.array([])
        self.__test_loss = np.array([])

        self.__train_acc = np.array([])
        self.__test_acc = np.array([])

        self.__avg_classification_time = 0.0
        self.__std_classification_time = 0.0

    def save_model(self, directory):
        current_time = datetime.today().strftime("%b-%d-%Y_%H-%M-%S")
        self.__model.save(directory + "/" + self.__finalized_model_filename + "_" + self.__variant + "_" + str(current_time) + ".sav")

    def load_model(self, path):
        self.__model = keras.models.load_model(path)

    def start_learning_process(self):
        log_message = f"*-----({self.__variant}_LEARNING_PROCESS_STARTED)-----*"
        self.__io.append_log(log_message)
        # learning_rates = [ 0.000001, 0.00001, 0.0001 ]

        # best_accuracy = 0.0
        # best_learning_rate = 0.0

        # for learning_rate in learning_rates:
        #     self.__train(variant = self.__variant, dataset = "train", learning_rate = learning_rate, epochs = 1000)
        #     (accuracy, f1) = self.__validate()
        #     log_message = f"(Validation) learning_rate = {learning_rate}; accuracy = {accuracy}; f1 = {f1}" 
        #     self.__io.append_log(log_message)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_learning_rate = learning_rate

        best_learning_rate = 0.0001

        start = datetime.now()
        self.__train(variant = self.__variant, dataset = "full", learning_rate = best_learning_rate, epochs = 1000)
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
        filename = f"{self.__variant}_metrics_results_{current_time}_{best_learning_rate}.txt"
        content = log_message
        self.__io.save_results("cnn", filename, content)

    def __initialize_resnet_model(self, variant = "resnet-34", learning_rate = 0.01):
        if self.__dirty == True:
            del self.__model
            self.__train_loss_avg = np.array([])
            self.__test_loss = np.array([])
            self.__train_acc = np.array([])
            self.__test_acc = np.array([])
            self.__avg_classification_time = 0.0
            self.__std_classification_time = 0.0
            self.__dirty = False

        match variant:
            case "resnet-18":
                self.__model = keras.models.Sequential()
                self.__model.add(keras.layers.Conv2D(64, 7, strides = 2, input_shape = [50, 50, 3], padding = "same", use_bias = False))
                self.__model.add(keras.layers.BatchNormalization())
                self.__model.add(keras.layers.Activation("relu"))
                self.__model.add(keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same"))
                prev_filters = 64
                for filters in [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2:
                    strides = 1 if filters == prev_filters else 2
                    self.__model.add(ResidualUnit(filters, strides = strides))
                    prev_filters = filters
                self.__model.add(keras.layers.GlobalAvgPool2D()) # or GlobalAveragePooling2D()
                self.__model.add(keras.layers.Flatten())
                self.__model.add(keras.layers.Dense(1, activation = "sigmoid")) # must be sigmoid
                self.__model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"])        

            case "resnet-34":
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
                self.__model.add(keras.layers.GlobalAvgPool2D()) # or GlobalAveragePooling2D()
                self.__model.add(keras.layers.Flatten())
                self.__model.add(keras.layers.Dense(1, activation = "sigmoid")) # must be sigmoid
                self.__model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.SGD(learning_rate = learning_rate), metrics = ["accuracy"]) 
            case "resnet-50":
                self.__model = ResNet50(learning_rate)
            case "resnet-101":
                self.__model = ResNet101(learning_rate)
            case "resnet-152":
                self.__model = ResNet152(learning_rate)
            case _:
                print("[ERROR] Such variant of ResNet architecture does not exist. Aborting further execution of the program.")
                sys.exit(1)


    def __train(self, variant = "resnet-34", dataset = "train", learning_rate = 0.01, epochs = 10):
        print("\n*--------------------TRAINING-CONVOLUTIONAL-NEURAL-NETWORK--------------------*")
        print(f"[INFO] Training of convolutional neural network model ({variant}) started.")

        gpu_devices = tf.config.list_physical_devices('GPU')
        print("[INFO] Number of GPUs available:", len(gpu_devices))

        self.__initialize_resnet_model(variant, learning_rate)

        # Loading 1000 image examples for testing
        first_positive_test_image = self.__training_set_positive_examples_count + self.__validation_set_positive_examples_count + 1 
        last_positive_test_image = first_positive_test_image + 500
        first_negative_test_image = self.__training_set_negative_examples_count + self.__validation_set_negative_examples_count + 1 
        last_negative_test_image = first_negative_test_image + 500

        # Load matrices of data
        (X_pos, y_pos) = self.__im.load_image_data(first_positive_test_image, last_positive_test_image, 1)
        (X_neg, y_neg) = self.__im.load_image_data(first_negative_test_image, last_negative_test_image, 0)

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
            (X_pos, y_pos) = self.__im.load_image_data(first_positive_image, last_positive_image, 1)
            (X_neg, y_neg) = self.__im.load_image_data(first_negative_image, last_negative_image, 0)

            X_training = np.concatenate((X_pos, X_neg), axis = 0)
            y_training = np.hstack([y_pos, y_neg])

            # Learning from batch
            history = self.__model.fit(X_training, y_training, epochs = epochs, shuffle = True, callbacks = [self.__callback])

            if dataset == "full":
                # Train loss
                self.__train_loss_avg = np.append(self.__train_loss_avg, statistics.mean(history.history["loss"]))  

            positive_images_processed += positive_images_batch_count
            negative_images_processed += negative_images_batch_count

            print("[INFO] Batch of", positive_images_batch_count + negative_images_batch_count, "examples --->", positive_images_batch_count, "positive and", negative_images_batch_count, "negative --->", positive_images_processed + negative_images_processed, "examples in total.")

        self.__dirty = True
        print("[INFO] Training done.")
        print("[INFO] Total images processed:", positive_images_processed + negative_images_processed)

        if dataset == "full":
            self.save_model("./saved_models")

    def __validate(self):
        print("\n*--------------------VALIDATING-CONVOLUTIONAL-NEURAL-NETWORK--------------------*")
        print(f"[INFO] Validation of convolutional neural network ({self.__variant}) model started.")
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
        print(f"[INFO] Validation of convolutional neural network ({self.__variant}) model finished.")

        return (accuracy[1], f1)

    def __test(self):
        print("\n*--------------------TESTING-CONVOLUTIONAL-NEURAL-NETWORK--------------------*")
        print(f"[INFO] Testing of convolutional neural network ({self.__variant}) model started.")
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

        tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

        # Time benchmark
        test_examples_count = self.__test_set_positive_examples_count + self.__test_set_negative_examples_count
        mean_classif_time = 0.0 # [s]
        std_classif_time = 0.0 # [s]
        classif_times = np.array([])

        count = 0
        for test_example in X_test:
            test_example = test_example[np.newaxis, ...]
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
        print("\n*--------------------CONVOLUTIONAL-NEURAL-NETWORK-TESTING-RESULTS--------------------*")
        print("[INFO] Accuracy obtained on the test set:", accuracy[1])
        print("[INFO] F1 score obtained on the test set:", f1)

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize = (10, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, y_predicted, display_labels = ["0", "1"], ax = ax)
        plt.show()
        fig.savefig(f"{self.__variant}_confusion_matrix.png")

        # ROC curve
        fig, ax = plt.subplots(figsize = (10, 8))
        fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='CNN (AUC = {:.2f})'.format(roc_auc), color = "b")
        plt.xlabel('False Positive Rate (Positive label: 1.0)')
        plt.ylabel('True Positive Rate (Positive label: 1.0)')
        plt.title('ROC curve')
        plt.legend(loc='lower right')
        plt.show()
        fig.savefig(f"{self.__variant}_cnn_roc.png")
        
        # Train loss
        x = np.linspace(1000, 1000 + self.__train_loss_avg.size * 1000, self.__train_loss_avg.size, endpoint = False)
        fig, ax = plt.subplots(figsize = (10, 8))
        plt.plot(x, self.__train_loss_avg, color = "b")
        plt.title(f"{self.__variant} model loss function value during training")
        plt.ylabel("Loss")
        plt.xlabel("Training examples")
        plt.legend(["Train loss"], loc = "upper right")
        plt.show()
        fig.savefig(f"{self.__variant}_cnn_loss.png")

        # Train and test accuracies
        x = np.linspace(1000, 1000 + self.__train_acc.size * 1000, self.__train_acc.size, endpoint = False)
        fig, ax = plt.subplots(figsize = (10, 8))
        plt.plot(x, self.__train_acc, color = "b")
        plt.plot(x, self.__test_acc, color = "r")
        plt.title(f"{self.__variant} model train and test accuracies during training")
        plt.ylabel("Accuracy")
        plt.xlabel("Training examples")
        plt.legend(["Train accuracy", "Test accuracy"], loc = "upper right")
        plt.show()
        fig.savefig('cnn_accuracy.png')

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


