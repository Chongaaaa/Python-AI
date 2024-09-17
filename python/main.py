import streamlit as st
import pandas as pd
import pickle

movies = pickle.load(open("../dataset/movies.pkl", "rb"))
cosine = pickle.load(open("../dataset/cosine_sim.pkl", "rb"))

# emoji icon link: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Mobie", page_icon=":movie_camera:", layout= "wide")

st.title("Welcome to MOBIE")

st.header("Movie Recommender System")
st.selectbox("Select movies from dropdown", movies)

# Cosine Similarity
def recommendMovie(movie_title, cosine_sim):
    # Check if the movie exists in the DataFrame
    if movie_title not in df['Series_Title'].values:
        print(f"Movie '{movie_title}' not found in the database.")
        return []

    # Find the index of the movie that matches the title
    index = df[df['Series_Title'] == movie_title].index[0]
    distance = sorted(list(enumerate(cosine_sim[index])), reverse=True, key=lambda vector: vector[1])

    # Get the top 10 most similar movies
    recommended_movies = [df.iloc[i[0]]['Series_Title'] 
    for i in distance[0:10]]

    return recommended_movies

# Display result
if st.button("Show Recommend"):
    pass