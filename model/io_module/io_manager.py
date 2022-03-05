from datetime import datetime
import os

class IOManager:
    def __init__(self):
        self.__output_dir = "./output"

    def save_results(self, directory, filename, content):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        if not os.path.isdir(self.__output_dir + "/" + directory):
            os.mkdir(self.__output_dir + "/" + directory)
        file_path = self.__output_dir + "/" + directory + "/" + filename
        with open(file_path, 'w+') as f:
            f.write(content)
