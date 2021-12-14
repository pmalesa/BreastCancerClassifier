from training.training_module import TrainingModule
import os

from PIL import Image
import numpy as np

from image_processing.image_manager import ImageManager

def main():
    tm = TrainingModule()
    tm.run()

    # im = ImageManager()
    # (X, y) = im.load_image_data(1, 1001, 1)
    # print("X:", X)
    # print("y:", y)
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)

    return 0

if __name__ == "__main__":
    main()



# Remarks:
# - Maybe refactor the training module class a little bit (create a function to load image data)
# - Think about a better solution to load and store data
# - Think how to change the way of shuffling the data (right now your method is very inefficient, because it relies on copying the matrices)

# - MAKE SURE THAT THE BATCHING PROCEDURE YOU CREATED WORKS PROPERLY - 





























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