import logging

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import TensorDataset


def prepare_dataset(device, dataset):
    logging.info("[DNN] Splitting dataset into X and y")
    df_X = dataset.drop('fraudRisk', axis=1)
    df_y = dataset[['fraudRisk']]

    logging.info("[DNN] Normalizing dataset...")
    df_X = (df_X - df_X.mean()) / df_X.std()

    X = df_X.values
    y = df_y.values

    logging.info("[DNN] Label Binarizer...")
    y = LabelBinarizer().fit_transform(y)

    # Test/train data split
    logging.info("[DNN] Test/train data split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=345, stratify=y)

    # Oversample only the training data
    logging.info("[DNN] Oversampling training data...")
    X_train, y_train = SMOTE(random_state=345).fit_resample(X_train, y_train)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return \
        TensorDataset(torch.tensor(X_train, requires_grad=True).to(device),
                      torch.tensor(y_train, requires_grad=True).unsqueeze(1).to(device)),\
        TensorDataset(torch.tensor(X_test, requires_grad=True).to(device),
                      torch.tensor(y_test, requires_grad=True).to(device))
