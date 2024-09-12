import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Asus/Desktop/Python/dataset/imdb_top_1000.csv")

movies_df = df.drop(columns = ['Certificate', 'Runtime', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Gross'])

movies_df['R_Score'] = movies_df['IMDB_Rating'] * 0.5 + movies_df['No_of_Votes'] * 0.5
movies_recommend_df = movies_df.sort_values(['R_Score'], ascending = False)

movies_recommend_df.head(8)