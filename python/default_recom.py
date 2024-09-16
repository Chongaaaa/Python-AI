from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

def generate_default_recommend():
    # Read the CSV file into a DataFrame
    df = pd.read_csv("C:/Users/Asus/Desktop/Python/dataset/imdb_top_1000.csv")
    
    # Drop unnecessary columns
    movies_df = df.drop(columns=['Certificate', 'Runtime', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Gross'])
    
    # Create a new column 'R_Score'
    movies_df['R_Score'] = movies_df['IMDB_Rating'] * 0.5 + movies_df['No_of_Votes'] * 0.5
    
    # Sort by 'R_Score' in descending order
    movies_recommend_df = movies_df.sort_values(['R_Score'], ascending=False)
    
    # Return the DataFrame
    return movies_recommend_df[['Poster_Link', 'Series_Title']]

@app.route('/')
def index():
    recommended_movies = generate_default_recommend()
    movies_data = recommended_movies.head(6).to_dict(orient='records')
    return render_template('index.html', movies_data=movies_data)

if __name__ == '__main__':
    app.run(debug=True)