import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data
movies = pd.read_csv("/dataset/movies.csv")

#select columns
movies = movies[["Series_Title","Overview", "Genre", "Director","Stars"]]

#combine columns
movies["genre_labels"] = movies["Genre"] + " " + movies["Overview"] + " " + movies["Director"] + " "  + movies["Stars"] 

#assign new variable to combine columns
movies_genre_parts = movies[["Series_Title", "genre_labels"]]
#------------------------------------------------------------------
#transform into matrix vector
from sklearn.feature_extraction.text import CountVectorizer
import pickle

cv = CountVectorizer(max_features=1000, stop_words='english')
cv_matrix = cv.fit_transform(movies_genre_parts["genre_labels"])
genre_vectors = cv_matrix.toarray()

#------------------------------------------------------------------
# Train a KNN model on the vectorized data
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors()
knn.fit(genre_vectors)

#Apply cosine similarity at the vectorized knn model
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(genre_vectors)

#--------------------------------------------------------------------
#Recommend and evaluate movies based on KNN model
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Assuming you have the similarity matrix
# Initialize PCA and KNN
pca = PCA(n_components=1000)  # Use the number of components you want
similarity_reduced = pca.fit_transform(similarity)  # Apply PCA to reduce dimensionality

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn.fit(similarity_reduced)

def get_true_relevant_movies(movie_title):
    """
    Dynamically retrieve relevant movies based on the provided movie title.
    For simplicity, this function will just return a list of similar movies
    from the dataset as the relevant movies.
    """
    # Find movies that are similar to the input movie title
    similar_movies = movies_genre_parts[movies_genre_parts["Series_Title"].str.contains(movie_title, case=False, na=False)]
    
    # Return the list of similar movie titles
    return similar_movies["Series_Title"].tolist()

def recommend_and_evaluate(movie, k=10):
    # Start the timer
    start_time = time.time()
    
    # Normalize user input
    movie = movie.strip().lower()
    
    # Find the closest matching movie
    matching_movies = movies_genre_parts[movies_genre_parts["Series_Title"].str.lower() == movie]
    
    if matching_movies.empty:
        # If no exact match is found, use partial matching
        matching_movies = get_true_relevant_movies(movie)
        if not matching_movies:
            print(f"No close match found for '{movie}'. Please check the spelling.")
            return
        closest_match = matching_movies[0]  # Just pick the first match
    else:
        closest_match = movie
    
    # Find the index of the closest matching movie
    try:
        movie_index = movies_genre_parts[movies_genre_parts["Series_Title"].str.lower() == closest_match].index[0]
    except IndexError:
        print(f"Movie '{closest_match}' not found in the dataset.")
        return
    
    # Reduce the dimensionality of the input movie (same transformation as training)
    movie_reduced = pca.transform(similarity[movie_index].reshape(1, -1))
    
    # Perform the k-nearest neighbors search
    distances, indices = knn.kneighbors(movie_reduced, n_neighbors=k)
    indices = indices.flatten()  # Flatten the indices to use for Pandas indexing
    
    # Convert indices to integer for Pandas indexing
    recommended_movies = movies_genre_parts.iloc[indices]["Series_Title"].values[:k]

    # Display the recommended movies
    print(f"Top {k} recommended movies for '{closest_match}':")
    for i, title in enumerate(recommended_movies):
        print(f"{i+1}: {title}")

    # Get true relevant movies based on the closest match
    true_relevant_movies = get_true_relevant_movies(closest_match)

    # Calculate Precision@k
    precision_at_k = len(set(recommended_movies) & set(true_relevant_movies)) / k
    print(f"Precision@{k}: {precision_at_k}")
    
    # Calculate Recall@k
    recall_at_k = len(set(recommended_movies) & set(true_relevant_movies)) / len(true_relevant_movies)
    print(f"Recall@{k}: {recall_at_k}")
    
    # Calculate F1 Score
    if precision_at_k + recall_at_k > 0:
        f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_score = 0
    print(f"F1 Score: {f1_score}")
    
    # End the timer
    end_time = time.time()
    print(f"Time taken to recommend: {end_time - start_time:.4f} seconds")

# Get user input for the movie name
user_movie = input("Please enter the movie name: ")

# Call the function with user input
recommend_and_evaluate(user_movie, k=5)
