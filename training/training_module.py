from sklearn.linear_model import LogisticRegression
import os


class TrainingModule:
    def __init__(self):
        self.__log_reg = LogisticRegression(max_iter = 1000)

        self.__output_file = "results.txt"
        self.__output_dir = "./output"
        self.__features = []

    def predict_class(self, X_new):
        y_prob = self.__log_reg.predict_proba(X_new)
        if y_prob[:, 1] >= 0.5:
            return 1
        else:
            return 0

    def perform_logistic_regression(self, X, y):
        self.__log_reg.fit(X, y)

    # TO DO
    def produce_results(self):
        if not os.path.isdir(self.__output_dir):
            os.mkdir(self.__output_dir)
        file_path = self.__output_dir + "/" + self.__output_file
        with open(file_path, 'w') as f:
            f.write("Results")