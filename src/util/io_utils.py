import os
import logging
import pandas as pd
from graphdatascience.error.unable_to_connect import UnableToConnectError

from src.models.DNN import DNN
from src.models.ForestClassifier import ForestClassifier
from src.neogds.NeoGDS import NeoGDS


def validate_digit_input(choice, lower_bound, upper_bound) -> bool:
    """
    Utility function to validate that a string satisfies these conditions:
    1. The string is a digit
    2. The string is lesser than the upper bound
    3. The string is greater than the lower bound
    :param choice: String to validate based on the aforementioned conditions.
    :param lower_bound: Lower bound for string validation.
    :param upper_bound: Upper bound for string validation.
    :return: Whether the given string passes the aforementioned criteria.
    """
    if not choice.isdigit() or int(choice) < lower_bound or int(choice) > upper_bound:
        logging.info(f"Invalid choice. Please enter a number from {lower_bound} to {upper_bound}.")
        return False
    return True


def print_choices() -> None:
    """
    Function that prints the menu options.
    :return: None
    """
    logging.info("""
Hello, welcome to my CS345 application.
In this application I use a graph bank fraud detection dataset.
This dataset has the possibility of having extra features computed
due to its graph nature.

You can perform these actions:
-----------------------------------------------------------------
        ---- RUN A MODEL ----
        1) Baseline           -> NO graph features,  NO DNN
        2) Classifier + Graph -> YES graph features, NO DNN
        3) Just DNN           -> NO graph features,  YES DNN
        4) DNN + Graph        -> YES graph features, YES DNN
        
        ---- OTHER ----
        5) Exit the program
-----------------------------------------------------------------
    """)


def read_choices(dataset_file_name, dataframe) -> None:
    """
    Function that reads and maps choices to their respective functions.
    :return: None
    """

    choice = input("Your choice: ")
    if not validate_digit_input(choice, 1, 6):
        return

    if dataframe is None:
        try:
            dataframe = load_dataframe(dataset_file_name)
        except UnableToConnectError:
            logging.error("""
            Unable to digest Neo4j dataframe on localhost:7687.
            This may be due to these reasons:
            ------------------------------------------------------------------------------
            a) You haven't started up Neo4j. Make sure to spin up a container using the provided docker-compose.yml.
            b) Another program is using the port 7687.
            c) You haven't setup Neo4j using the import-fraud-database.bash script.
            """)
            quit(1)

    if choice == "1":
        ForestClassifier().evaluate(partialize_dataframe(dataframe))
    elif choice == "2":
        ForestClassifier().evaluate(dataframe)
    elif choice == "3":
        df = partialize_dataframe(dataframe)
        DNN(df.shape[1] - 1).evaluate(df)
    elif choice == "4":
        DNN(dataframe.shape[1] - 1).evaluate(dataframe)
    elif choice == "5":
        logging.info("""
                This program has ceased to be! It is but an ex-program!
                    _
                  /` '\\
                /| @   l
                \\|      \\
                  `\\     `\\_
                    \\    __ `\\
                    l  \\   `\\ `\\__
                     \\  `\\./`     ``\\
                       \\ ____ / \\   l
                         ||  ||  )  /
                -------(((-(((---l /-------
                                l /
                               / /
                              / /
                             //
                            /
                """)
        quit(0)  # Exit with a happy error code :)


def load_dataframe(dataset_file_name: str):
    dataset_exists = os.path.isfile(dataset_file_name)

    if dataset_exists:
        df = pd.read_pickle(dataset_file_name)
        logging.info(f"[MAIN] Bank dataset file {dataset_file_name} imported.")
    else:
        logging.info(f"[MAIN] {dataset_file_name} was NOT found. Creating it from Neo4j...")
        logging.info(f"[MAIN] Please be patient, this is a one-time fix up...")
        df = NeoGDS().get_df()
        df.to_pickle(dataset_file_name)
        logging.info(f"[MAIN] Saved \"{dataset_file_name}\".")

    return df


def partialize_dataframe(df):
    return df[['fraudRisk', 'numberOfDevices', 'numberOfCCs', 'numberOfIps',
               'totalOutgoingAmount', 'maxOutgoingAmount', 'avgOutgoingAmount',
               'totalIncomingAmount', 'maxIncomingAmount', 'avgIncomingAmount',
               'outgoingTransactions', 'incomingTransactions']]
