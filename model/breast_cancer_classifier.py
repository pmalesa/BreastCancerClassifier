from training.training_module import TrainingModule

def main():
    tm = TrainingModule("svm")
    tm.run()
    #tm.save_chosen_model()

    return 0

if __name__ == "__main__":
    main()



# Remarks:
# - Think how to change the way of shuffling the data (right now your method is very inefficient, because it relies on copying the matrices)
# - think about getting rid of ImageManager and IOManager objects from each model
# - use grid search for cross validation and finding the optimal hyperparameters
# - Think about how to omit the situation with declaring the model object with parameters in the constructor and then calling reset at the
#   beginning of the train function which initializes the object once again
# - Think about using the scikit learn's grid search class, but it is not necessary

# - SOMETHING IS WRONG WITH ITERATING OVER THE HYPERPARAMETERS IN SVM MODEL CLASS, AND IN THE OTHERS PROBABLY TOO
# - CROSSVALIDATE THE CNN WITH THE ARGUMENTS PASSED TO FIT FUNCTION


























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