from image_processing.image_manager import ImageManager
from training.training_module import TrainingModule
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pickle

def main():
    positive_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/1/"
    negative_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/0/"

    n = 60000
    nAttributes = 7500
    checkpoint = 1000

    tm = TrainingModule()
    im = ImageManager()

    # ---------------------------------------------------------------------------------
    #                                       LEARNING
    # ---------------------------------------------------------------------------------

    nPositiveImages = im.get_number_of_files(positive_images_directory)
    nNegativeImages = im.get_number_of_files(negative_images_directory)

    y_positive = np.ones(n)
    y_negative = np.zeros(n)

    print("Positive images:", nPositiveImages)
    print("y_positive size:", y_positive.shape)
    print("Negative images:", nNegativeImages)
    print("y_negative size:", y_negative.shape)

    pixel_colors_matrix_pos = np.zeros((n, nAttributes), dtype = float)

    # Load positive examples
    for filenumber in range(1, n + 1):
        filepath = positive_images_directory + str(filenumber) + ".png"
        pixel_colors_vec = im.get_pixel_colors(filepath)
        pixel_colors_matrix_pos[filenumber - 1, :] = pixel_colors_vec
        if filenumber % checkpoint == 0:
            print("Loaded", filenumber, "positive images.")

    pixel_colors_matrix_neg = np.zeros((n, nAttributes), dtype = float)

    # Load negative examples
    for filenumber in range(1, n + 1):
        filepath = negative_images_directory + str(filenumber) + ".png"
        pixel_colors_vec = im.get_pixel_colors(filepath)
        pixel_colors_matrix_neg[filenumber - 1, :] = pixel_colors_vec
        if filenumber % checkpoint == 0:
            print("Loaded", filenumber, "negative images.")    

    # Concatenate positive and negative matrices
    X = np.vstack([pixel_colors_matrix_pos, pixel_colors_matrix_neg])
    y = np.hstack([y_positive, y_negative])

    print("X size:", X.shape)
    print("y size:", y.shape)

    tm.perform_logistic_regression(X, y)

    print("Learning done.")

    # ---------------------------------------------------------------------------------
    #                                    LEARNING DONE
    # ---------------------------------------------------------------------------------
    
    finalized_model_filename = "finalized_model.sav"

    # ---------------------------------------------------------------------------------
    # Save model to the disk
    pickle.dump(tm, open(finalized_model_filename, "wb"))

    # Load the model from the disk
    # tm = pickle.load(open(finalized_model_filename, "rb"))
    # ---------------------------------------------------------------------------------

    # Testing
    X_new_pos = np.array(im.get_pixel_colors(positive_images_directory + str(n + 100) + ".png"))
    X_new_neg = np.array(im.get_pixel_colors(negative_images_directory + str(n + 100) + ".png"))

    X_new_pos = X_new_pos[np.newaxis, :]
    X_new_neg = X_new_neg[np.newaxis, :]

    y_1 = tm.predict_class(X_new_pos)
    y_2 = tm.predict_class(X_new_neg)

    print("Result of prediction on positive class image", str(n + 100) + ".png:", y_1)
    print("Result of prediction on negative class image", str(n + 100) + ".png:", y_2)

    return 0

if __name__ == "__main__":
    main()










# ---------------------------------------------------------------------------------
# for filenumber in range(2, 21):
#     filepath = positive_images_directory + str(filenumber) + ".png"
#     pixel_colors_vec = im.get_pixel_colors(filepath)
#     pixel_colors_matrix = np.vstack([pixel_colors_matrix, pixel_colors_vec])
#     if filenumber % 1000 == 0:
#         print("Loaded", filenumber, "images.")
# ---------------------------------------------------------------------------------
# DELETING WRONGLY SIZED IMAGES (THESE WHICH ARE NOT 50x50)
# positive_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/1/"
# negative_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/0/"
# im = ImageManager()
# im.delete_wrong_sized_images(positive_images_directory)
# im.delete_wrong_sized_images(negative_images_directory)
# ---------------------------------------------------------------------------------
# CATEGORIZING IMAGES
# im = ImageManager()
# src_dir = "../../Praca_Inzynierska/Breast_Histopathology_Images/"
# dest_dir = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/"
# im.divide_into_categories(src_dir, dest_dir)
# ---------------------------------------------------------------------------------
# READING PIXELS FORM PNG FILE
# im = ImageManager()
# pixels = im.get_pixel_vector("./images/test_1.png")
# for pixel in pixels:
    # print(pixel)
    # break
# ---------------------------------------------------------------------------------
# CONVERTING TO BMP
# im = ImageManager()
# src_dir = "../../Praca_Inzynierska/Breast_Histopathology_Images/"
# dest_dir = "../../Praca_Inzynierska/Breast_Histopathology_Images_BMP/"
# im.convert_files_to_bmp(src_dir, dest_dir)
# ---------------------------------------------------------------------------------