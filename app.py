import streamlit as st
import pandas as pd

import requests
import json

st.title("Qurram's CSCI5901 project dashboard")

tab1, tab2 = st.tabs(["Interface", "Monitoring"])

with tab1:
    st.header("Model interface")

    status_dict = list(requests.get("http://ec2-3-89-36-216.compute-1.amazonaws.com:8501/v1/models/model").json().values())[0][0]
    if status_dict['state'] == 'AVAILABLE':
        st.markdown(f'<p style="color:white;font-size:24px;border-radius:2%;width:120px;border:1px solid '
                    f'#09ab3b;text-align:center;background-color:#09ab3b;border-radius:5px">AVAILABLE</p>',
                    unsafe_allow_html=True)

    number = st.number_input('Insert index for entry in test dataset (0-56985)', step=1, max_value=56985, min_value=0)
    if number:
        dataset = pd.read_csv('survival_processed.csv')
        dataset.drop('Unnamed: 0', axis=1, inplace=True)
        dataset.drop('hospital_death', axis=1, inplace=True)

        random_entry = dataset.iloc[int(number)]

        data = json.dumps({"signature_name": "serving_default", "instances": random_entry.values.reshape(1, -1).tolist()})

        headers = {"content-type": "application/json"}
        json_response = requests.post('http://ec2-3-89-36-216.compute-1.amazonaws.com:8501/v1/models/model:predict', data=data, headers=headers)
        print("Response", json_response.json())
        pred = json_response.json()["predictions"][0][0]
        st.markdown(f'<span style="color:#a3a8b4;">Probability of hospital death: </span> <span style=color:{"#ff2b2b" if pred > 0.5 else "#09ab3b"}>{pred}</span>', unsafe_allow_html=True)
        st.header('Selected entry: ')
        st.dataframe(random_entry)