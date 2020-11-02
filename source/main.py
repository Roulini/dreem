import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import features as feat 
import train_model as train_model
import joblib
import os


f_train = h5py.File("data/raw/X_train.h5/X_train.h5","r")
f_test = h5py.File("data/raw/X_test.h5/X_test.h5","r")


y_train_raw = pd.read_csv('data/raw/y_train.csv')

def train():
    X_train=feat.get_features(f_train)

    y_train_vals = y_train_raw["sleep_stage"].values

    # on passe l'output en chaine de caracteres 
    y_train_vals = [str(y) for y in y_train_vals]

    #train model and save results
    train_model.train_full_model(X_train,y_train_vals)

#predict data_test
def predict():
    X_test = feat.get_features(f_test,100)
    scaler = joblib.load("models/scaler1")
    model = joblib.load("models/model1.save")
    x_test_scaled = scaler.transform(X_test)
    y_predicted = model.predict(x_test_scaled)
    output = pd.DataFrame({"index": f_test["index_absolute"][:], "sleep_stage": y_predicted})
    output.to_csv("output2.csv",index=False)
    print("over")

predict()