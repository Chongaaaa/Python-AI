import streamlit as st
import pandas as pd
import base64

# Load the dataset
df = pd.read_csv(r'C:/Users/Asus/Desktop/Python/dataset/imdb_top_1000.csv')

# Set page configuration
st.set_page_config(page_title="Mobie", page_icon=":movie_camera:", layout="wide")

# Optionally, print the DataFrame
st.dataframe(df)
