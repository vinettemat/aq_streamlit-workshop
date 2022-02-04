import pandas as pd
import streamlit as st
from streamlit import write
from transformers import pipeline
import os
from google.cloud import firestore

def app():
    db = firestore.Client.from_service_account_info(st.secrets["gcp_service_account"])
    st.write("This is my first app:") #Affichage d'un titre dans l'app
    unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
    sentence = st.text_input('Fill in the sentence you want to try then press enter:', 'Data science is [MASK].')
    if "[MASK]" in sentence:
      result = unmasker(sentence)
      st.write(pd.DataFrame(result))
    else:
      st.warning("The sentence needs to contains [MASK]")



if __name__ == '__main__':

    app()
