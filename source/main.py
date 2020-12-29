import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import features as feat 
import train_model as train_model
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
import time


f_train = h5py.File("data/raw/X_train.h5/X_train.h5","r")
f_test = h5py.File("data/raw/X_test.h5/X_test.h5","r")


y_train_raw = pd.read_csv('data/raw/y_train.csv')

def train():
    n_pts = None #change to test on a subset of the data
    X_features=feat.get_features(f_train,n_pts)
    joblib.dump(X_features,"models/X_features.save")
    if n_pts!= None :
        y_train_vals=y_train_raw["sleep_stage"][:n_pts].values
    else :
        y_train_vals = y_train_raw["sleep_stage"][:].values

    # on passe l'output en chaine de caracteres 
    y_train_vals = [str(y) for y in y_train_vals]

    #train model and save results
    print("training")
    train_model.train_full_model(X_features,y_train_vals)



#predict data_test
def predict():
    i=5
    X_test = feat.get_features(f_test)
    scaler = joblib.load("models/scaler"+str(i)+".save")
    model = joblib.load("models/model"+str(i)+".save")
    x_test_scaled = scaler.transform(X_test)
    joblib.dump(x_test_scaled,"models/x_to_pred_scaled"+str(i)+".save")
    y_predicted = model.predict(x_test_scaled)
    output = pd.DataFrame({"index": f_test["index_absolute"][:], "sleep_stage": y_predicted})
    output.to_csv("output"+str(i)+".csv",index=False)
    print("over")

def train_new_model():
    #train model with the 
    i = 5
    x_train_scaled = joblib.load("models/x_train"+str(i)+".save")
    y_train = joblib.load("models/y_train"+str(i)+".save")
    x_test_scaled = joblib.load("models/x_to_pred_scaled"+str(i)+".save")
    scaler = joblib.load("models/scaler"+str(i)+".save")

    # model = joblib.load("models/model"+str(i)+".save")
    
    model = GradientBoostingClassifier(n_estimators=2000,max_features = 30, max_depth = 4)
    print("train")
    tic = time.time()
    model.fit(x_train_scaled,y_train)
    print("elapsed time", time.time()-tic)

    joblib.dump(model,"models/model_retrain"+str(i)+".save")

    y_predicted = model.predict(x_test_scaled)
    output = pd.DataFrame({"index": f_test["index_absolute"][:], "sleep_stage": y_predicted})
    output.to_csv("output_retrain"+str(i)+".csv",index=False)


# train()
# predict()
train_new_model()



