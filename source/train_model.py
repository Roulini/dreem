from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

def save_data(model,scaler,x,y):
    i=2
    joblib.dump(model,"models/model"+str(i)+".save")
    joblib.dump(x,"models/x_train"+str(i)+".save")
    joblib.dump(y,"models/y_train"+str(i)+".save")
    joblib.dump(scaler,"models/scaler"+str(i)+".save")
    

def train_model_and_score(X,y_train):
    """train model for test"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    #chose model
    model = RandomForestClassifier()

    #split train/test
    x_train,x_test,y_train,y_test = train_test_split(X_scaled,y_train,test_size=0.33,random_state =42)

    #train
    model.fit(x_train,y_train)

    #evaluation
    sc = model.score(x_test,y_test), model.score(x_train,y_train)

    print(sc)

    return model,sc

def train_full_model(X,y_train):
    """train model with all the training models for kaggle submission"""
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X)

    #chose model
    model = RandomForestClassifier()

    #train
    model.fit(x_train,y_train)

    #save data
    save_data(model,scaler,x_train,y_train)

