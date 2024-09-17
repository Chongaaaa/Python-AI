import streamlit as st
import pickle

movies= pickle.load(open('movies.pkl', 'rb'))
st.selectbox("Select Movie", movies)

movies_list=movies['Series_Title'].values
