from training.training_module import TrainingModule

def main():
    tm = TrainingModule("cnn")
    tm.run()

    return 0

if __name__ == "__main__":
    main()