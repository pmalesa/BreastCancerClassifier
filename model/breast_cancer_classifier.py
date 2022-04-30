from training.training_module import TrainingModule

def main():
    tm = TrainingModule("cnn")
    tm.run()
    # tm.save_chosen_model()

    return 0

if __name__ == "__main__":
    main()



# Remarks:
# - DATA AUGMENTATION - Implement rotating each image in the loaded batch (if batch size is 1000 then you will have 4000 examples in total), and use all of the created images
#   to train the model + some gaussian noise or random noise (https://www.kaggle.com/code/zeadomar/breast-cancer-detection-with-cnn)
# - Deal with the biased dataset, either think about those weights for each class (Weight balancing), or use different approach,
# - Expand the ResNet-34 model, so that maybe it will perform better (maybe use ResNet-18 before the main model as Nathan suggested).
# - Also, maybe think about those number of steps within each epoch (one step corresponds to one batch of loaded samples, so if your batch is 1000, then it also can be splitted
#   into smaller batches - by default you have 32 steps per each epoch, which means that your batch size is 32 within each 'bigger' batch - see the keras documentation for more).

























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