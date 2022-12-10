""" Contains the model classes used for temperature prediction. """

import numpy as np

import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    """ LSTM model to make outside temperature predictions. """

    output_size: int

    hidden_layer_size: int
    lstm: nn.LSTM
    linear: nn.Linear
    hidden_cell: tuple[torch.Tensor, torch.Tensor]
    scaler: MinMaxScaler
    device: str

    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()

        self.output_size = output_size

        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)


        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

    def forward(self, input_seq):
        """ Foward function. """
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def predict(self, input):
        """ Prediction function. """

        X = self.scaler.fit_transform(input.reshape(-1, 1))
        X = seq = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            self.hidden = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))
            preds = self(X)
            Y = np.array(preds.cpu())
        actual_predictions = self.scaler.inverse_transform(Y.reshape(-1, 1))
        return actual_predictions

    def set_device(self):
        """ Sets the device to cuda if available, or cpu if not.
        SHOULD ALWAYS BE RAN WHEN LOADING FROM PICKLE!! """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self.to(self.device)

    def train(self, train_data, train_window = 50, epochs = 100):
        """ Simple train function that computes a defined number of epochs. """

        # Initiate loss function and optimizer

        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Scale

        train_data_normalized = self.scaler.fit_transform(train_data.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1).to(self.device)

        # Create sequences
        
        train_inout_seq = []
        L = len(train_data_normalized)
        for i in range(L-train_window):
            train_seq = train_data_normalized[i:i+train_window]
            # TODO - offset on the start for direct decoding with several (5) of these models giving (5) concurrent predictions?
            train_label = train_data_normalized[i+train_window:i+train_window+self.output_size]
            train_inout_seq.append((train_seq ,train_label))

        # Train the model

        for i in range(epochs):
            _ix = -1
            for seq, labels in train_inout_seq:
                seq = seq.to(self.device)
                labels = labels.to(self.device)

                if len(labels) != self.output_size:
                    continue

                _ix +=1

                #if torch.isnan(seq).any().item():
                #    print(f"nan values in seq at {_ix}")
                #    continue

                #if pd.isna(labels.item()):
                #    print(f"nan labels at {_ix}")
                #    continue

                optimizer.zero_grad()
                self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                                torch.zeros(1, 1, self.hidden_layer_size).to(self.device))

                y_pred = self(seq)
                #if pd.isna(y_pred.item()):
                #    print(f"nan preds at {_ix}")
                #    continue

                single_loss = loss_function(y_pred, labels)
                #if pd.isna(single_loss.item()):
                #    print(f"nan loss at {_ix}")
                #    raise
                single_loss.backward()
                optimizer.step()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        # Finalize
        #self.eval()

class CustomRandomForestRegressor(RandomForestRegressor):
    """ Model used to make inside temperature predictions.
    Is depedent on the outside predictions. """

    n_preds: int
    n_estimators: int

    def __init__(self, n_preds=4,n_estimators=100):
        super().__init__(n_estimators=100)
        self.n_preds = n_preds
        self.n_estimators = n_estimators

    def total_predict(self, outside_val_curr: float, outside_val_preds: np.array, inside_val_curr: float):
        """ Makes all the sequential predictions of the defined int n_preds values. """
        inside_preds = []
        assert len(outside_val_preds) == self.n_preds
        for i in range(self.n_preds):
            if i == 0:
                input = np.array([outside_val_curr, outside_val_preds[i][0], inside_val_curr], dtype="object").reshape(1, -1)
            else:
                input = np.array([outside_val_preds[i-1], outside_val_preds[i][0], inside_preds[i-1]], dtype="object").reshape(1, -1)
            pred = self.predict(input)
            inside_preds.append(pred)
        return inside_preds
