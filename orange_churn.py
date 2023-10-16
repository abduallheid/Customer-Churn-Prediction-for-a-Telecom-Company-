# load important libraries
import pandas as pd 
import numpy as np 
import joblib 
import streamlit as st 
import time
from sklearn.preprocessing import StandardScaler
st.info(body="Developed by Eng: Abduallh Eid")
st.code(body="Gmail: abduallheid87@gmail.com")

# Load the Model 
model= joblib.load("RF_v1.ml")

st.header("Orange Telecommunications Churn Customer App")
# image and text
st.image("orange.png",caption="Orange Telecom churn Model")
st.text(body="The Orange Telecom's Churn Dataset, which consists of cleaned\n customer activity data (features), along with a churn label\n specifying whether a customer canceled the subscription,\n will be used to develop predictive models")


def upload_data():

    # function csv file upload
    file =st.file_uploader("Upload file csv consisting of customer activity data")
    def upload (file):  
        st = st.open(file  + ".csv")  
        df=pd.read_csv(file + ".csv")
        st.dataframe(df.head(10)) 
                
    def show(file): 
        df=pd.read_csv(file)
        st.dataframe(df.head(10))

    if st.button("Show sample data"):  # show state
        show(file)

    def forcast(file):

        df=pd.read_csv(file)
                   
        # handle object and bool col
        for col in df.columns:
            if df[col].dtype=="object":
                df[col]=df[col].replace({"yes":1,"no":0})
            elif df[col].dtype=="bool":
                df[col]=df[col].astype(int)

        SC=StandardScaler()
        Sdata=SC.fit_transform(df) # scaled dataset
        df["churned"]=0    # creat churn column
        for i in range(0, 75): #adding forcast records
            pred=model.predict(Sdata[i:i+1])
                        
            df.iloc[i:i+1,-1:]=pred
        # compression_opts=dict(method="zip",archive_name="forcast.csv")
        # df.to_csv("forcast.zip",index=False, compression=compression_opts)
        st.download_button("download",df.to_csv(index=False))    
    if st.button("download predicted records csv"):
        bar=st.progress(30)
        time.sleep(2)
        bar.progress(100)
        forcast(file)
        



# function  add values  manually
def manual():

    # assign input parameters 
    ACC=st.number_input(label="account lenth ",max_value=300)
    INTLP=st.number_input("intl",min_value=0 ,max_value=1)
    voice= st.number_input("voice intl plan",min_value=0 ,max_value=1)
    vmail=st.number_input("number_vmail_messages",min_value=0 ,max_value=80)
    tdm=st.number_input("total_day_minutes",min_value=0 ,max_value=355)
    tdc=st.number_input("total_day_calls",min_value=0 ,max_value=170)
    tdch=st.number_input("total_day_charge",min_value=0 ,max_value=70)
    tem=st.number_input("total_eve_minutes",min_value=0 ,max_value=363)
    tecal=st.number_input("total_eve_calls",min_value=0 ,max_value=200)
    tec=st.number_input("total_eve_charge",min_value=0 ,max_value=40)
    tnm=st.number_input("total_night_minutes",min_value=0 ,max_value=178)
    tnc=st.number_input("total_night_calls",min_value=0 ,max_value=180)
    tnch=st.number_input("total_night_charge",min_value=0 ,max_value=30)
    tim=st.number_input("total_intl_minutes",min_value=0 ,max_value=400)
    tical=st.number_input("total_intl_calls",min_value=0 ,max_value=20)
    tich=st.number_input("total_intl_charge",min_value=0 ,max_value=20)
    ncs=st.number_input("number_customer_service_calls",min_value=0 ,max_value=100)

    # transform features to an array
    feat=np.array([ACC,INTLP,voice,vmail,tdm,tdc,tdch,tem,tecal,tec,tnm,tnc,tnch,tim,tical,tich,ncs])

    res =model.predict(feat.reshape(1,-1))


    if  st.button("predict"):
        if res==[1]:
            st.success(f"Your emp is churn")
        elif res!=[1]:
            st.success(f"Your emp isnt churn")

    download_data= pd.Series(feat+res)
    download_data =download_data.to_csv(index=False)    
    st.download_button("download",download_data+".csv")    
# choice the way you want    
but1 = st.button("ADD features manually")
but2 = st.button("upload features csv files")
but1,but2= st.columns([2,1])
if but2:
    upload_data()
elif but1:
    manual()
st.columns()
      
