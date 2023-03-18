import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.util.dataset_utils import prepare_dataset


# =========================================================================================================================
class DNN(nn.Module):
    # --------------------------------------------------------------------------------------
    def __init__(self, input_features):
        super(DNN, self).__init__()

        # .................... Parameter Config ........................
        self.feature_count = input_features
        # ..............................................................

        # .................... Training Config .........................
        self.batch_size = 8192
        self.training_epochs = 2
        self.learning_rate = 0.05
        self.cost_function = nn.BCELoss()
        # ..............................................................

        # ...................... Layer Config ..........................
        self.lin1 = (nn.Linear(in_features=self.feature_count, out_features=8, bias=True))
        self.relu1 = (nn.ReLU())
        self.batch1 = (nn.BatchNorm1d(num_features=8))

        self.lin2 = (nn.Linear(in_features=8, out_features=16, bias=True))
        self.relu2 = (nn.ReLU())
        self.batch2 = (nn.BatchNorm1d(num_features=16))

        self.lin3 = (nn.Linear(in_features=16, out_features=32, bias=True))
        self.relu3 = (nn.ReLU())
        self.batch3 = (nn.BatchNorm1d(num_features=32))

        self.lin4 = (nn.Linear(in_features=32, out_features=64, bias=True))
        self.relu4 = (nn.ReLU())
        self.batch4 = (nn.BatchNorm1d(num_features=64))

        self.lin5 = (nn.Linear(in_features=64, out_features=64, bias=True))
        self.relu5 = (nn.ReLU())
        self.batch5 = (nn.BatchNorm1d(num_features=64))

        self.out = (nn.Linear(in_features=64, out_features=1, bias=True))
        self.sigmoid = (nn.Sigmoid())
        # ..............................................................

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Init optimizer after params were moved to the GPU
        self.optimizer = SGD(params=self.parameters(), lr=self.learning_rate, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", min_lr=0.001, patience=1)

    def forward(self, dataset):
        # Feed forward to the next layer
        ta = dataset
        ta = self.lin1(ta)
        ta = self.relu1(ta)
        ta = self.batch1(ta)

        ta = self.lin2(ta)
        ta = self.relu2(ta)
        ta = self.batch2(ta)

        ta = self.lin3(ta)
        ta = self.relu3(ta)
        ta = self.batch3(ta)

        ta = self.lin4(ta)
        ta = self.relu4(ta)
        ta = self.batch4(ta)

        ta = self.lin5(ta)
        ta = self.relu5(ta)
        ta = self.batch5(ta)

        ta = self.out(ta)
        ta = self.sigmoid(ta)
        return ta
        # --------------------------------------------------------------------------------------

    def train_epoch(self, training_loader):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self(inputs)

            # Compute the loss and its gradients
            loss = self.cost_function(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.batch_size == self.batch_size - 1:
                last_loss = running_loss / self.batch_size  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        return last_loss

    def training_loop(self, dataset):
        dataset_train, dataset_test = dataset

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

        # Training Loop
        epoch_number = 0
        best_val_loss = 1_000_000.
        for epoch in range(self.training_epochs):
            print(f'EPOCH {epoch_number + 1}:')

            # Make sure gradient tracking is on, and do a pass over the data
            self.train(True)
            avg_loss = self.train_epoch(training_loader)

            # We don't need gradients on to do reporting
            self.train(False)

            running_val_loss = 0.0
            index = 0
            for index, val_data in enumerate(validation_loader):
                val_inputs, val_labels = val_data
                val_outputs = self(val_inputs)
                val_loss = self.cost_function(val_outputs, val_labels.float())
                running_val_loss += val_loss

            avg_val_loss = running_val_loss / (index + 1)
            print('LOSS train {} valid {} next lr {}'.format(avg_loss, avg_val_loss, self.optimizer.param_groups[0]['lr']))

            # Adjust learning rate if necessary
            self.scheduler.step(running_val_loss)

            # Track best performance, and save the model's state
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                # torch.save(model.state_dict(), model_path)
            epoch_number += 1

    @torch.no_grad()
    def predict(self, X_test):
        y_test = self(X_test)
        return y_test.cpu()

    def evaluate(self, dataset):
        # Prep the dataset
        dataset = prepare_dataset(self.device, dataset)

        # Train model
        logging.info("[DNN] Starting training...")

        self.training_loop(dataset)

        # Predict values
        y_pred = self.predict(dataset[1].tensors[0])

        # Evaluate the model
        logging.info("[DNN] Evaluated model, displaying stats.")

        true = np.array(dataset[1].tensors[1].detach().cpu()).astype(np.float)
        pred = np.array(y_pred.round()).astype(np.float)

        # Make the confusion matrix
        ConfusionMatrixDisplay.from_predictions(true, pred)
        RocCurveDisplay.from_predictions(true, pred)
        plt.show()

# =========================================================================================================================
