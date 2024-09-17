import streamlit as st
import pandas as pd
import pickle

movies = pickle.load(open("../dataset/movies.pkl", "rb"))

# emoji icon link: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Mobie", page_icon=":movie_camera:", layout= "wide")

st.title("Welcome to MOBIE")

st.header("Movie Recommender System")
st.selectbox("Select movies from dropdown", movies)

# Cosine Similarity


if st.button("Show Recommend"):
    pass