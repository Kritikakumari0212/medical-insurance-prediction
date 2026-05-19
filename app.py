import streamlit as st
import numpy as np
import joblib
model=joblib.load('model.pkl')
encoder=joblib.load('Hot.pkl')

#container to display prediction results
result=st.empty()

st.title("INSURANCE CHARGES PREDICTION")
st.write("Enter customer details below to predict insurance charges.")
#for user input
age=st.number_input("Enter your age")
sex=st.selectbox("Select Gender",['male','female'])
bmi=st.number_input("Enter your BMI")
children=st.number_input("No. of childrens you have")
smoker=st.radio("Are you a smoker?",['yes','no'])
region=st.selectbox("Enter your region",
                    ['southwest',
                     'northwest',
                     'southeast',
                     'northeast'])

num_data=[age,bmi,children]   #numerical features
if st.button('Predict'):
    num_data=[[age,bmi,children]]

    cat_data=[[sex,smoker,region]]

    encodeData=encoder.transform(cat_data) #encode categorical data

    #convert matrix to array
    #encodeData=encodeData.toarray()

    #combine numerical and categorical data

    finalOutput=np.concatenate((num_data,encodeData),axis=1)

    #prediction
    yPred=model.predict(finalOutput)

    #display result
    result.success(f"Estimated Insurance Charge is : rs. {yPred[0]:.2f}")
    