from pprint import pprint
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from ptImplementation import Data, NeuralNetwork
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
from datetime import datetime
from torch import nn, optim
import itertools
from sklearn.preprocessing import MinMaxScaler
class MidasDataProcessing():

    def __init__(self, linearFeaturesIncluded=True):
        self.linearFeaturesIncluded = linearFeaturesIncluded
        if self.linearFeaturesIncluded:
            self.availableFeatures = ['ob_time', 'wind_speed', 'wind_direction', 'cld_ttl_amt_id', 'cld_base_ht_id_1', 'visibility', 'msl_pressure', 'air_temperature', 'rltv_hum']
        else:
            self.availableFeatures = ['ob_time', 'wind_speed', 'wind_direction', 'cld_ttl_amt_id', 'cld_base_ht', 'visibility', 'msl_pressure', 'air_temperature']


    def fetchData(self, location='heathrow'):
        # print('DATA FETCHED')
        df2020 = pd.read_csv(os.getcwd()+'/Data/MIDAS/'+location+'_weather_2020.csv', skiprows =280, skipfooter=1, engine='python')
        rain_2020 = pd.read_csv(os.getcwd()+'/Data/MIDAS/'+location+'_rain_2020.csv', skiprows =61, skipfooter=1, engine='python')
        df2020 = df2020[self.availableFeatures]
        df2020['prcp_amt'] = rain_2020['prcp_amt']

        df2021 = pd.read_csv(os.getcwd()+'/Data/MIDAS/'+location+'_weather_2021.csv', skiprows =280, skipfooter=1, engine='python')
        rain_2021 = pd.read_csv(os.getcwd()+'/Data/MIDAS/'+location+'_rain_2021.csv', skiprows =61, skipfooter=1, engine='python')
        df2021 = df2021[self.availableFeatures]
        df2021['prcp_amt'] = rain_2021['prcp_amt']

        df2022 = pd.read_csv(os.getcwd()+'/Data/MIDAS/'+location+'_weather_2022.csv', skiprows =280, skipfooter=1, engine='python')
        rain_2022 = pd.read_csv(os.getcwd()+'/Data/MIDAS/'+location+'_rain_2022.csv', skiprows =61, skipfooter=1, engine='python')
        df2022 = df2022[self.availableFeatures]
        df2022['prcp_amt'] = rain_2022['prcp_amt']

        df = pd.concat([df2020, df2021, df2022], ignore_index=True)
        df = df.dropna(axis=1, how='all')

        # Convert str to datetime
        df['ob_time'] = pd.to_datetime(df ['ob_time'], format='%Y-%m-%d %H:%M:%S')
        for col in df.columns:
            df.rename(columns={col: location+' '+col}, inplace=True)

        return df

    def convertDatestoYearlyFloat(self, mainLocation, df):
        # convert datetime to float
        # df['Date'] = [datetime.timestamp(x) for x in df['ob_time']]

        # Remove leap year day and change all years to 2019 so tm_yday is consistent.
        df = df[~((df[mainLocation+' ob_time'].dt.month == 2) & (df[mainLocation+' ob_time'].dt.day == 29))]
        df[mainLocation+' ob_time'] = [x.replace(year=2020) for x in df[mainLocation+' ob_time']]
        hours = [x.hour for x in df[mainLocation+' ob_time']]
        df['Date'] = [x.timetuple().tm_yday+(y*1/24) for x,y in zip(df[mainLocation+' ob_time'], hours)]
        df = df.drop([mainLocation+' ob_time'], axis=1)
        # df.rename(columns={'air_temperature': 'y'}, inplace=True)
        # self.dfFuture = df[-2000:]
        # df = df[:-2000]
        return df

    def cleanDf(self, df):
        df = df.dropna(axis=0, how='any')
        return df


    def feature_label_split(self, df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(self, df, target_col, test_ratio):
        val_ratio = test_ratio / (1 - test_ratio)
        X, y = self.feature_label_split(df, target_col)
        self.all_x = X.to_numpy()
        self.all_y = y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def datasplit(self,df, target_col):
        # y = np.array(df['y'])
        # X = np.array(df[self.features])
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.33, random_state=26)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.train_val_test_split(df, target_col, 0.2)
        scaler = MinMaxScaler()
        self.unNormalisedX_train = self.X_train.copy()
        self.unNormalisedX_test = self.X_test.copy()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

        # self.y_train = scaler.fit_transform(np.array(self.y_train[target_col]).reshape(-1,1))
        # self.y_val = scaler.transform(np.array(self.y_val[target_col]).reshape(-1,1))
        # self.y_test = scaler.transform(np.array(self.y_test[target_col]).reshape(-1,1))

        self.y_train = np.array(self.y_train[target_col]).reshape(-1,1)
        self.y_val = np.array(self.y_val[target_col]).reshape(-1,1)
        self.y_test = np.array(self.y_test[target_col]).reshape(-1,1)

        train_data = TensorDataset(torch.tensor(self.X_train), torch.tensor(self.y_train))
        train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
        val_data = TensorDataset(torch.tensor(self.X_val), torch.tensor(self.y_val))
        val_dataloader = DataLoader(dataset=val_data, batch_size=64, shuffle=True)
        test_data = TensorDataset(torch.tensor(self.X_test), torch.tensor(self.y_test))
        test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
        test_loader_one = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=True)
        train_loader_one = DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader, train_loader_one, test_loader_one,

    def plotData(self, df):
        plt.rcParams["figure.figsize"] = (20,5)
        fontsize=14
        fig, axs = plt.subplots(ncols=1, nrows=len(self.trainingFeatures), figsize=(30,30))
        axes = 0
        # df = df.iloc[0::100]
        for j, f1 in enumerate(self.trainingFeatures):
            # for i, f1 in enumerate(df.columns):
                f2 = 'y'
                ax = fig.get_axes()[axes]
                ax.scatter(df[f1],df[f2],  s=3)
                # if f1 == 'msl_pressure':
                #     ax.set_xscale('log')
                if axes<8:
                    ax.set_title(f1, fontsize=fontsize)
                if axes <56:
                    ax.set_xticklabels([])
                if axes%8==0:
                    ax.set_ylabel(f2, fontsize=fontsize)
                else:
                    ax.set_yticklabels([])
                axes +=1
                if f1 == 'dewpoint' and f2 == 'y':
                    ax.plot([-20,20], [-20,20], color='red')
                    # fig.get_axes()[8].scatter(df['dewpoint'], df['rltv_hum'], s=3)
        # fig.savefig('Figures/RNNPredictionMIDAS.pdf')
        plt.show()

    def addSpatialFeatures(self, mainLocation, mainLocationdf, location, spatialFeatures):

        subLocationDf = self.fetchData(location)
        for f in spatialFeatures:
            mainLocationdf[location+' '+f] = mainLocationdf[mainLocation+' ob_time'].map(subLocationDf.set_index(location+' ob_time')[location+' '+f])
            # mainLocationdf[location+' wind_speed'] = mainLocationdf[mainLocation+' ob_time'].map(subLocationDf.set_index(location+' ob_time')[location+' wind_speed'])
            # mainLocationdf[location+' air_temperature'] = mainLocationdf[mainLocation+' ob_time'].map(subLocationDf.set_index(location+' ob_time')[location+' air_temperature'])

        print(mainLocationdf.columns)

        return mainLocationdf
        # self.plotData(locationDf)

    def createSpatialDf(self, mainLocation='keswick'):
        temporalDf = self.fetchData(mainLocation)
        print(temporalDf.columns)
        if self.linearFeaturesIncluded:
            spatialFeatures = ['wind_speed', 'wind_direction', 'msl_pressure', 'dewpoint', 'rltv_hum']
        else:
            spatialFeatures = ['wind_speed', 'wind_direction', 'msl_pressure']
        names = ['st-bees-head-no-2', 'warcop-range', 'shap']
        for n in names:
            temporalDf = self.addSpatialFeatures(mainLocation, temporalDf, n, spatialFeatures)

        for name in names:
            for feature in spatialFeatures:
                temporalDf[name+' '+feature] = temporalDf[name+' '+feature].shift(1)

        # print(len(temporalDf))
        temporalDf = self.convertDatestoYearlyFloat(mainLocation, temporalDf)
        temporalDf = self.cleanDf(temporalDf)
        # print(temporalDf.head())
        # print(len(temporalDf))

        self.allFeatures = temporalDf.columns
        self.trainingFeatures = [x for x in self.allFeatures if x!= (mainLocation+' air_temperature')]
        self.inputDim = len(self.allFeatures)-1
        # self.plotLocations(names)

        return temporalDf

    def create_temporal_df(self, mainLocation='heathrow'):
        temporalDf = self.fetchData(mainLocation)
        temporalDf = self.convertDatestoYearlyFloat(mainLocation, temporalDf)
        temporalDf = self.cleanDf(temporalDf)
        self.allFeatures = temporalDf.columns
        self.trainingFeatures = [x for x in self.allFeatures if x!= (mainLocation+' air_temperature')]
        self.inputDim = len(self.allFeatures)-1
        return temporalDf
