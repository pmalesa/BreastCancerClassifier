from image_processing.image_manager import ImageManager
from training.training_module import TrainingModule
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

def main():
    im = ImageManager()
    pixel_colors_1 = np.array(im.get_pixel_colors("./images/test_1.png"))
    print("Image 1 RGB pixel values:")
    print(pixel_colors_1)

    pixel_colors_2 = np.array(im.get_pixel_colors("./images/test_2.png"))
    print("Image 2 RGB pixel values:")
    print(pixel_colors_2)

    print("Both images' RGB pixel values together:")
    pixel_colors_combined = np.array([pixel_colors_1, pixel_colors_2])
    print(pixel_colors_combined)

    # ---------------------------------------------------------------------------------
    # CATEGORIZING IMAGES
    # fm = FileManager()
    # src_dir = "../../Praca_Inzynierska/Breast_Histopathology_Images/"
    # dest_dir = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/"
    # fm.divide_into_categories(src_dir, dest_dir)
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

    return 0

if __name__ == "__main__":
    main()