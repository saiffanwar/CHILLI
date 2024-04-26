import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from MidasDataProcessing import MidasDataProcessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import matplotlib.lines as mlines


is_cuda = torch.cuda.is_available()
#is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.ones(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, h0 = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'saved/models/MIDAS_model.pck'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                if x_batch.shape[0] == batch_size:
                    x_batch, y_batch = x_batch.to(torch.float32), y_batch.to(torch.float32)
                    x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                    y_batch = y_batch.to(device)
                    loss = self.train_step(x_batch, y_batch)
                    batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    if x_val.shape[0] == batch_size:
                        x_val, y_val = x_val.to(torch.float32), y_val.to(torch.float32)

                        x_val = x_val.view([batch_size, -1, n_features]).to(device)
                        y_val = y_val.to(device)
                        self.model.eval()
                        yhat = self.model(x_val)
                        val_loss = self.loss_fn(y_val, yhat).item()
                        batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                x_test, y_test = x_test.to(torch.float32), y_test.to(torch.float32)
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(float(yhat))
                values.append(float(y_test))

        return predictions, values

    def predictor_from_numpy(self, x_test, batch_size=2000):
        batch_size = len(x_test)
        x_test = torch.from_numpy(np.array(x_test))
        x_test = (x_test.to(torch.float32)).reshape([batch_size, -1, len(x_test[0])])
        pred = self.model(x_test)
        floatpreds = np.array([float(p) for p in pred])

        return floatpreds


