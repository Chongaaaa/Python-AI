import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Load Data
df = pd.read_csv("C:/Users/Asus/Desktop/Python/dataset/movies.csv")

df = df.drop(columns = ["Unnamed: 0"])
df["Movies_Infor"] = df["Released_Year"].astype(str) + " " + df["Genre"] + " " + df["Overview"] + " " + df["IMDB_Rating"].astype(str) + " " + df["Director"] + " " + df["Stars"]

x = df["Movies_Infor"]
y = df["Series_Title"]


# ----------------------------------
# convert movie information into numeric
tfidf_vectorizer = TfidfVectorizer(max_features = 1000)
x_tfidf = tfidf_vectorizer.fit_transform(x)
y_tfidf = tfidf_vectorizer.fit_transform(y)

# Apply LSA to reduce dimension
lsa = TruncatedSVD(n_components = 100)
x_lsa = lsa.fit_transform(x_tfidf)
y_lsa = lsa.fit_transform(y_tfidf)


# ----------------------------------
# Algorithm Calculation
# Use euclidean distance to find the nearest 
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def find_similar_movies(target_vec, lsa):
    distances = []
    for idx, movie_vec in enumerate(lsa):
        diff = euclidean_distance(target_vec, movie_vec)
        distances.append((idx, diff))
    
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])
    
    # Return similar movies
    return distances[:10]


# ----------------------------------
# Link the movies name here from drop down
user_movie = input("Enter a movie name: ")

movie_index = df[df["Series_Title"] == user_movie].index[0]

# Perform similarity search on the full dataset
target_vec = x_lsa[movie_index]
similar_movies = find_similar_movies(target_vec, x_lsa)

# Link display session at here
for i, score in similar_movies:
    print("{}: {}".format(i, df.loc[i, "Series_Title"]))


# ----------------------------------
# check the efficiency and accuraccy
same_series = df[df["Series_Title"].str.contains(user_movie, case=False, na=False)]

# Precision: measures how many of the top recommended items are relevant
def precision_at_k(y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))  # Intersection of ground truth and predicted
    if relevant == same_series.shape[0]: 
        return relevant / same_series.shape[0]
    return relevant / k

# Recall: measures how many relevant items are captured in the top recommendation
def recall_at_k(y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))
    if relevant == same_series.shape[0]: 
        return relevant / same_series.shape[0]
    return relevant / k

# harmonic mean of precision and recall
def f1_at_k(y_true, y_pred, k):
    prec = precision_at_k(y_true, y_pred, k)
    rec = recall_at_k(y_true, y_pred, k)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

# actual movie
y_true = same_series["Series_Title"]

# Predict movie    
y_pred = [df.loc[i, "Series_Title"] for i, score in similar_movies]

k = 10
precision = precision_at_k(y_true, y_pred, k)
recall = recall_at_k(y_true, y_pred, k)
f1 = f1_at_k(y_true, y_pred, k)

#print(f"Precision@{k}: {precision:.2f}") # AKA: Accurancy
#print(f"Recall@{k}: {recall:.2f}")
#print(f"F1@{k}: {f1:.2f}")
#print("Predicted movies: ", y_pred)