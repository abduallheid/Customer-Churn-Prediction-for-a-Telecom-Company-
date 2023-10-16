
# import libaraies and frameworks
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score ,classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score



# set warnings for compiler 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# path = "dataset/Orange_Telecom_Churn_Data.csv"

# function which show data 


def read_data(path ):   
    data=pd.read_csv(path)
    return print(data.head(10))

# preprocessing data 

def pre_data(data):
    
    # drop unused columns 
    data.drop(columns=["state","area_code","phone_number"],inplace=True)
    
    # handle object and bool col
    for col in data.columns:
        if data[col].dtype=="object":
            data[col]=data[col].replace({"yes":1,"no":0})
        elif data[col].dtype=="bool":
            data[col]=data[col].astype(int)
    # float columns         
    float_col = [x for x in data.columns if data[x].dtype=="float64"]
    return float_col

# train test split data for modeling 

def train_tst_split(data):
    
    X=data.drop(columns="churned")
    y=data["churned"]
    x_trin ,x_test ,y_train ,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    return x_trin ,x_test ,y_train ,y_test

# pipeline function Automating Machine Learning with Pipelines

""" RF= RandomForestClassifier(oob_score=True, 
                            random_state=42, 
                            warm_start=True,
                            n_estimators=100,
                            n_jobs=-1)"""
def pipeline_RF (estimator ,scaler ,x_trin ,x_test ,y_train ,y_test ):
    
    
    sc = scaler 
    Model = estimator
    
    # bulit pipeline 
    estimators =[("scaler",sc),("RF":Model)]
    
    pipe =pipeline(estimators)
    
    # fit scaler & model 
    
    pipe=pipe.fit(x_trin,y_train)
    
    y_pred = pipe.predict(x_test)
    
    # accuarcy metrics 
    
    ## Preciision, recall, f-score
    precision, recall, fscore, _ = score(y_test,y_pred, average='weighted')
    # the usuall accuracy
    accuracy = accuracy_score(y_test,y_pred)
    # ROC_AUC 
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1]),
              label_binarize(y_pred, classes=[0,1]), 
              average='weighted')
    
    return precision, recall, fscore , auc
        
# save model joblib         
def save_model (estimator):
    
    joblib.dump(Model,"model_v1.ml") # save model as name  "model_v1.ml"
                
    
    
    




              