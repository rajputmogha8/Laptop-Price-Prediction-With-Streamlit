#import the model
import pickle
import sklearn
import numpy as np
df=pickle.load(open('df.pkl','rb'))
pipe2 = pickle.load(open('pipe.pkl','rb'))
import streamlit as st
st.title("Laptop Predictions")
#Brand Name
Company=st.selectbox('Brand',df['Company'].unique())
Type=st.selectbox('Type',df['TypeName'].unique())
Ram=st.number_input('Ram')
Weight=st.number_input('Weight')
Touchscreen=st.selectbox('Touchscreen',['Yes','No'])
Ips=st.selectbox('Ips',['Yes','No'])
screensize  = st.number_input('Screensize_inch')
Resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
CpuBrand=st.selectbox('CpuBrand',df['Cpu brand'].unique())
HDD=st.number_input('HDD')
SDD=st.number_input('SDD')
Gpubrand=st.selectbox('Gpu brand',df['Gpu brand'].unique())
OS=st.selectbox('os',df['os'].unique())
if st.button('Predict Price'):
    # query
    ppi = None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0

    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))*0.5/screensize
    query = np.array([Company,Type,Ram,Weight,Touchscreen,Ips,ppi,CpuBrand,HDD,SDD,Gpubrand,OS])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe2.predict(query)[0]))))
