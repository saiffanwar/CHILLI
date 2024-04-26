import midas_model_data as midas
import torch.optim as optim
import torch
from torch import nn
import numpy as np
import pickle as pck
import json


from MidasDataProcessing import MidasDataProcessing
from chilli import CHILLI, chilli_explain

is_cuda = torch.cuda.is_available()
is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class MIDAS():

    def __init__(self, load_model=False):
        self.load_model = load_model
        self.data = MidasDataProcessing(linearFeaturesIncluded=True)
        cleanedFeatureNames = ['Heathrow wind speed', 'Heathrow wind direction', 'Heathrow total cloud cover', 'Heathrow cloud base height', 'Heathrow visibility', 'Heathrow MSL pressure', 'Date']
        self.df = self.data.create_temporal_df(mainLocation='heathrow')
        self.train_loader, self.val_loader, self.test_loader, self.train_loader_one, self.test_loader_one = self.data.datasplit(self.df, 'heathrow air_temperature')

    def train_midas_rnn(self,):

        self.input_dim = self.data.inputDim
        self.output_dim = 1
        self.hidden_dim = 12
        self.layer_dim = 3
        self.batch_size = 64
        self.dropout = 0.2
        self.n_epochs = 200
        self.learning_rate = 1e-2
        self.weight_decay = 1e-6

        model_params = {'input_dim': self.input_dim,
                        'hidden_dim' : self.hidden_dim,
                        'layer_dim' : self.layer_dim,
                        'output_dim' : self.output_dim,
                        'dropout_prob' : self.dropout}

        model_path = f'saved/models/MIDAS_model.pck'
        self.model = midas.RNNModel(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim, self.dropout)

        if self.load_model:
            self.model.load_state_dict(torch.load(model_path))
        loss_fn = nn.L1Loss(reduction="mean")
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.opt = midas.Optimization(model=self.model.to(device), loss_fn=loss_fn, optimizer=optimizer)

        if not self.load_model:
            self.opt.train(self.train_loader, self.val_loader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.input_dim)
#            opt.plot_losses()
            torch.save(self.model.state_dict(), model_path)

        return self.opt.predictor_from_numpy

    def make_midas_predictions(self,):
        train_pred, values = self.opt.evaluate(self.train_loader_one, batch_size=1, n_features=self.input_dim)
        test_pred, values = self.opt.evaluate(self.test_loader_one, batch_size=1, n_features=self.input_dim)

        self.train_preds = np.array(train_pred).flatten()
        self.test_preds = np.array(test_pred).flatten()

        return self.train_preds, self.test_preds


if __name__ == '__main__':
    feature_types = {
    'Euclidean' : ['heathrow cld_ttl_amt_id', 'heathrow cld_base_ht', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow y', 'heathrow dewpoint', 'heathrow rltv_hum', 'heathrow wind_speed', 'heathrow air_temperature', 'heathrow prcp_amt'],

    'Cyclic': ['Date', 'wind_direction']
    }
    with open("feature_types.json", "w") as outfile:
        json.dump(feature_types, outfile)

    # Load the MIDAS class which does data preprocessing and loads model structure.
    midas_runner = MIDAS(load_model=True)

    x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures
    target_feature = 'heathrow air_temperature'
    categorical_features = ['heathrow cld_ttl_amt_id']

    # Train the MIDAS model
    model = midas_runner.train_midas_rnn()
    # Make predictions
    train_preds, test_preds = midas_runner.make_midas_predictions()

    instance_index = 50
    categorical_features = ['heathrow cld_ttl_amt_id']

    chilli_explain(model, x_train, y_train, train_preds, x_test, y_test, test_preds, features, target_feature, instance=instance_index, kernel_width=None, categorical_features=categorical_features)

