import shutil
import os

from shutil import copyfile
from os import walk
from PIL import Image

class ImageManager:
    def __init__(self):
        self.__nPositiveCopied = 0
        self.__nNegativeCopied = 0
        self.__nConvertedImages = 0

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
            width, height = im.size
            if width != 50 or height != 50:
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
            width, height = im.size
            if width != 50 or height != 50:
                os.remove(path)

    def get_pixel_colors(self, image):
        im = Image.open(image)
        width, height = im.size
        if width != 50 or height != 50:
            return []

        pixels = im.getdata()
        pixel_colors = []

        # 64-bit floats used for better performance
        for (r, g, b) in pixels:
            pixel_colors.append(float(r))
            pixel_colors.append(float(g))
            pixel_colors.append(float(b))
        return pixel_colors

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
