import shutil
import os

from shutil import copyfile
from os import walk
from PIL import Image
import numpy as np

class ImageManager:
    def __init__(self):
        self.__positive_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/1/"
        self.__negative_images_directory = "../../Praca_Inzynierska/Breast_Histopathology_Images_Categorized/0/"

        self.__positive_examples_count = self.get_number_of_files(self.__positive_images_directory)
        self.__negative_examples_count = self.get_number_of_files(self.__negative_images_directory)

        self.__image_height = 50
        self.__image_width = 50
        self.__image_channels = 3
        self.__attributes_count = 7500

        self.__nPositiveCopied = 0
        self.__nNegativeCopied = 0
        self.__nConvertedImages = 0

    def get_positive_examples_count(self):
        return self.__positive_examples_count
    
    def get_negative_examples_count(self):
        return self.__negative_examples_count

    def get_number_of_files(self, dir):
        path, dirs, files = next(os.walk(dir))
        return len(files)
    
    def save_wrong_sized_images(self, dir, output):
        f = []
        
        # Get directory and file names
        for (dirpath, dirnames, filenames) in walk(dir):
            f.extend(filenames)
            break

        for filename in filenames:
            im = Image.open(dir + filename)
            if im.size != (50, 50):
                with open(output, 'a+') as f:
                    f.write(filename + "\n")

    def delete_wrong_sized_images(self, dir):
        f = []
        
        # Get directory and file names
        for (dirpath, dirnames, filenames) in walk(dir):
            f.extend(filenames)
            break

        for filename in filenames:
            path = dir + filename
            im = Image.open(path)
            if im.size != (50, 50): 
                os.remove(path)

    def get_pixel_colors(self, image):
        im = Image.open(image)
        if im.size != (50, 50):       
            return []
        pixel_colors = np.asarray(im, dtype = float) / 255.0
        im.close()
        return pixel_colors

    # Loads data of images from first_image to last_image (last_image excluded) of given image_class
    def load_image_data(self, first_image, last_image, image_class):
        if first_image < 1 or last_image < 1:
            return ([], [])
        if first_image > last_image:
            return ([], [])
        if image_class == 1 and (first_image > self.__positive_examples_count or last_image > self.__positive_examples_count + 1):
            return ([], [])
        if image_class == 0 and (first_image > self.__negative_examples_count or last_image > self.__negative_examples_count + 1):
            return ([], [])
        
        image_count = last_image - first_image
        if image_class == 1:
            y = np.ones(image_count)
        else:
            y = np.zeros(image_count)
        X = np.zeros((image_count, self.__image_height, self.__image_width, self.__image_channels), dtype = float)

        example_index = 0
        for filenumber in range(first_image, last_image):
            if image_class == 1:
                filepath = self.__positive_images_directory + str(filenumber) + ".png"
            else:
                filepath = self.__negative_images_directory + str(filenumber) + ".png"
            X[example_index, :] = self.get_pixel_colors(filepath)
            example_index += 1
        return (X, y)

    def get_pixel_colors_flattened(self, image):
        im = Image.open(image)
        if im.size != (50, 50):        
            return []
        pixel_colors = np.asarray(im.getdata(), dtype = float).flatten() / 255.0
        im.close()
        return pixel_colors

    # Loads flattened data of images from first_image to last_image (last_image excluded) of given image_class
    def load_image_data_flattened(self, first_image, last_image, image_class):
        if first_image < 1 or last_image < 1:
            return ([], [])
        if first_image > last_image:
            return ([], [])
        if image_class == 1 and (first_image > self.__positive_examples_count or last_image > self.__positive_examples_count + 1):
            return ([], [])
        if image_class == 0 and (first_image > self.__negative_examples_count or last_image > self.__negative_examples_count + 1):
            return ([], [])
        
        image_count = last_image - first_image
        if image_class == 1:
            y = np.ones(image_count)
        else:
            y = np.zeros(image_count)
        X = np.zeros((image_count, self.__attributes_count), dtype = float)

        example_index = 0
        for filenumber in range(first_image, last_image):
            if image_class == 1:
                filepath = self.__positive_images_directory + str(filenumber) + ".png"
            else:
                filepath = self.__negative_images_directory + str(filenumber) + ".png"
            X[example_index, :] = self.get_pixel_colors_flattened(filepath)
            example_index += 1
        return (X, y)

    def divide_into_categories(self, src_dir, dest_dir):
        self.__nPositiveCopied = 0        
        self.__nNegativeCopied = 0

        # Create destination subdirectories
        dest_dir_positive = dest_dir + "/1"
        dest_dir_negative = dest_dir + "/0"

        if os.path.isdir(dest_dir_positive):
            shutil.rmtree(dest_dir_positive)
        if os.path.isdir(dest_dir_negative):
            shutil.rmtree(dest_dir_negative)

        try:
            os.mkdir(dest_dir_positive)
            os.mkdir(dest_dir_negative)
        except OSError as error:
            print(error)
            return

        self.__divide_into_categories_recursively(src_dir, dest_dir)
        total = self.__nPositiveCopied + self.__nNegativeCopied

        os.system("clear")
        print("Copied", self.__nPositiveCopied, "positive examples")
        print("Copied", self.__nNegativeCopied, "negative examples")
        print("Copied", total, "files in total.") 

    def __divide_into_categories_recursively(self, src_dir, dest_dir):
        f = []
        
        # Get directory and file names
        for (dirpath, dirnames, filenames) in walk(src_dir):
            f.extend(filenames)
            break
        
        # Copy to category destination directories
        category_dir = os.path.basename(src_dir)
        if category_dir == "0":
            dest_subdir = dest_dir + "/0"
            self.__copy_categorized_files(src_dir, dest_subdir)

        elif category_dir == "1":
            dest_subdir = dest_dir + "/1"
            self.__copy_categorized_files(src_dir, dest_subdir)

        # Recursively copy directories' contents
        for dirname in dirnames:
            new_src_dir = src_dir + "/" + dirname
            self.__divide_into_categories_recursively(new_src_dir, dest_dir)

    def __copy_categorized_files(self, src_dir, dest_dir):
        f = []

        # Get directory and file names
        for (dirpath, dirnames, filenames) in walk(src_dir):
            f.extend(filenames)
            break

        # Check if the opened subdirectory is a category subdirectory ("1" or "0")
        category_subdir = os.path.basename(src_dir)
        if category_subdir != "1" and category_subdir != "0":
            return

        for filename in filenames:
            full_src_path = src_dir + "/" + filename
            full_dest_path = dest_dir + "/"
            extension = os.path.splitext(filename)[1]
            if category_subdir == "1":
                full_dest_path += str(self.__nPositiveCopied + 1) + extension
                self.__nPositiveCopied += 1
            elif category_subdir == "0":
                full_dest_path += str(self.__nNegativeCopied + 1) + extension
                self.__nNegativeCopied += 1

            copyfile(full_src_path, full_dest_path)

            total = self.__nPositiveCopied + self.__nNegativeCopied
            if total % 10000 == 0:
                os.system("clear")
                print("Copied", total, "files so far.")

    def convert_files_to_bmp(self, dir_in, dir_out):
        self.nConvertedImages_ = 0
        self.__convert_recursively_to_bmp(dir_in, dir_out)
        os.system('clear')
        print("Converted", self.nConvertedImages_, "images in total.")

    def __convert_recursively_to_bmp(self, dir_in, dir_out):
        f = []
        
        # Get directory and file names
        for (dirpath, dirnames, filenames) in walk(dir_in):
            f.extend(filenames)
            break

        # Convert files
        for file in filenames:
            base = os.path.splitext(file)[0]
            self.convert_png_to_bmp(dir_in + file, dir_out + base + ".bmp")
            self.nConvertedImages_ += 1          
            if self.nConvertedImages_ % 10000 == 0:
                os.system('clear')
                print("Converted", self.nConvertedImages_, "images so far.")

        # Recursively convert directories' contents
        for dir in dirnames:
            src_dir = dir_in + dir + "/"
            dest_dir = dir_out + dir + "/"
            try:
                os.mkdir(dest_dir)
            except OSError as error:
                print(error)
            self.__convert_recursively_to_bmp(src_dir, dest_dir)

    def convert_png_to_bmp(self, file_in, file_out):
        img = Image.open(file_in)
        try:
            img.save(file_out)
        except OSError as error:
            print(error)

    def is_png(self, filename):
        try:
            image_fd = open(filename, "rb")
            image_bytes = bytearray(image_fd.read())
            image_fd.close()
            if self.is_png(image_bytes):
                return image_bytes
            else:
                return bytearray()
        except FileNotFoundError:
            print(f"File \"{filename}\" not found.")
            return bytearray()
        else:
            print(f"Unexpected error while opening \"{filename}\"")
            return bytearray()

        image_bytes = bytearray()

        if len(image_bytes) < 8:
            return False
        if image_bytes[0] != 0x89:
            return False 
        if image_bytes[1] != 0x50:
            return False
        if image_bytes[2] != 0x4E:
            return False
        if image_bytes[3] != 0x47:
            return False
        if image_bytes[4] != 0x0D:
            return False
        if image_bytes[5] != 0x0A:
            return False
        if image_bytes[6] != 0x1A:
            return False
        if image_bytes[7] != 0x0A:
            return False
        else:
            return True 
