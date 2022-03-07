from datetime import datetime
import os

import time

class IOManager:
    def __init__(self):
        self.__output_dir = "./output"
        self.__log_file = "log"

    def save_results(self, directory, filename, content):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        if not os.path.isdir(self.__output_dir + "/" + directory):
            os.mkdir(self.__output_dir + "/" + directory)
        file_path = self.__output_dir + "/" + directory + "/" + filename
        with open(file_path, 'w+') as f:
            f.write(content)

    def append_log(self, log_message):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        file_path = self.__output_dir + "/" + self.__log_file
        current_time = datetime.today().strftime("%b-%d-%Y %H:%M:%S")
        with open(file_path, 'a+') as f:
            f.write("[" + current_time + "] " + log_message + "\n")
