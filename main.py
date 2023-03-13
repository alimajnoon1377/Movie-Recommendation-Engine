import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
users = pd.read_csv('users.csv')
from sklearn.neighbors import NearestNeighbors

def build_user_based_cf_model(ratings):
    # pivot the ratings dataframe to get a user-item matrix
    user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

    # fill in missing values with 0
    user_item_matrix.fillna(0, inplace=True)

    # build a k-NN model using cosine similarity
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix)

    return model_knn, user_item_matrix
from flask import Flask, request, render_template
import pandas as pd
from recommender import build_user_based_cf_model, recommend_movies

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    userId = int(request.form['userId'])
    num_movies = int(request.form['numMovies'])

    # load the data and build the model
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    model, user_item_matrix = build_user_based_cf_model(ratings)

    # get the recommended movies
    recommendations = recommend_movies(model, user_item_matrix, userId, num_movies, movies)

    return render_template('recommend.html', userId=userId, recommendations=recommendations)

if __name__ == '__main__':
    app.run()
