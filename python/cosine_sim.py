import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load the dataset
df = pd.read_csv("/dataset/movies.csv")
df = df[['Series_Title', 'Genre', 'Overview', 'Director', 'Stars']]

# Combine relevant movie information into a single column
df['Movie_Info'] = df['Genre'] + ' ' + df['Overview'] + ' ' + df['Director'] + ' ' + df['Stars']

# Drop the now redundant columns
df = df.drop(columns=['Genre', 'Overview', 'Director', 'Stars'])

# Convert movie information into vectors
cv = CountVectorizer(max_features=1000, stop_words='english')
vector = cv.fit_transform(df['Movie_Info'].values.astype('U')).toarray()

# Calculate cosine similarity
cosine_sim = cosine_similarity(vector)

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

# Get user input
user_movie = input("Enter a movie name: ")

start_time = time.time()


# Display the recommended movies
recommended_movies = recommendMovie(user_movie, cosine_sim)
print(f"Top 10 movies similar to '{user_movie}':")
for movie in recommended_movies:
    print(movie)

end_time = time.time()

# Evaluate performance
same_series = df[df["Series_Title"].str.contains(user_movie, case=False, na=False)]

# Accuracy: the fraction of relevant recommendations
def accuracy_at_k(y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))
    return relevant / len(y_true)

# Precision: measures how many of the top recommended items are relevant
def precision_at_k(y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))  # Intersection of ground truth and predicted
    return relevant / k

# Recall: measures how many relevant items are captured in the top recommendation
def recall_at_k(y_true, y_pred, k):
    relevant = len(set(y_true) & set(y_pred[:k]))
    return relevant / len(y_true)

# Harmonic mean of precision and recall
def f1_at_k(y_true, y_pred, k):
    prec = precision_at_k(y_true, y_pred, k)
    rec = recall_at_k(y_true, y_pred, k)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)



# Actual movie titles that are considered similar (ground truth)
y_true = same_series["Series_Title"].tolist()

# Predicted movie titles from the recommendation
y_pred = recommended_movies

k = 10
accuracy = accuracy_at_k(y_true, y_pred, k)
precision = precision_at_k(y_true, y_pred, k)
recall = recall_at_k(y_true, y_pred, k)
f1 = f1_at_k(y_true, y_pred, k)

print(f"Accuracy@{k}: {accuracy:.2f}")
print(f"Precision@{k}: {precision:.2f}") 
print(f"Recall@{k}: {recall:.2f}")
print(f"F1@{k}: {f1:.2f}")
print(f"Executed Time: {end_time-start_time:.4f} seconds")