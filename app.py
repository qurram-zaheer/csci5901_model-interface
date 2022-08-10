import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import requests
import json
import os

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

with tab2:
    st.header("Model monitoring")

    list_dfs = {x: pd.read_csv(os.path.join('monitoring', x)) for x in os.listdir('monitoring')}
    
    first_dicts = {
        'monitoring_cpu.csv': 'CPU usage',
        'monitoring_memory.csv': 'Memory usage',
        'training_test_batch_accuracy.csv': 'Test batch accuracy',
        'training_test_batch_loss.csv': 'Test batch loss',
        'training_test_epoch_accuracy.csv': 'Test epoch accuracy',
    }

    second_dicts = {
        'training_test_epoch_loss.csv': 'Test epoch loss',
        'training_train_batch_accuracy.csv': 'Train batch accuracy',
        'training_train_batch_loss.csv': 'Train batch loss',
        'training_train_epoch_accuracy.csv': 'Train epoch accuracy',
        'training_train_epoch_loss.csv': 'Train epoch loss'
    }


    col1, col2 = st.columns(2)

    with col1:
        for k, v in first_dicts.items():
            fig, ax = plt.subplots()
            ax.plot(list_dfs[k].iloc[:,1], list_dfs[k].iloc[:,2])
            ax.set_title(v)
            st.pyplot(fig)

    with col2:
        for k, v in second_dicts.items():
            fig, ax = plt.subplots()
            ax.plot(list_dfs[k].iloc[:,1], list_dfs[k].iloc[:,2])
            ax.set_title(v)
            st.pyplot(fig)



    

