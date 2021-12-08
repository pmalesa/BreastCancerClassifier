import os

class TrainingModule:
    def __init__(self):
        self.__output_file = "results.txt"
        self.__output_dir = "./output"
        self.__features = []

    def produce_results(self):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        file_path = self.__output_dir + "/" + self.__output_file
        with open(file_path, 'w') as f:
            f.write("Results")