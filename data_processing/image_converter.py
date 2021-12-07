import os
from PIL import Image
from os import walk

class ImageConverter:
    def __init__(self):
        self.nConvertedImages_ = 0

    def convert_files_to_bmp(self, dir_in, dir_out):
        self.nConvertedImages_ = 0
        self.convert_recursively_to_bmp(dir_in, dir_out)
        os.system('clear')
        print("Converted", self.nConvertedImages_, "images in total.")

    def convert_recursively_to_bmp(self, dir_in, dir_out):
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
            self.convert_recursively_to_bmp(src_dir, dest_dir)

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
