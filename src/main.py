import logging

from src.util.io_utils import print_choices, read_choices

dataset_file_name = "bank-fraud-detection-ds.cs345"
dataframe = None


def run_app():
    global dataframe
    while True:
        print_choices()
        read_choices(dataset_file_name, dataframe)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.root.setLevel(logging.INFO)
    run_app()
