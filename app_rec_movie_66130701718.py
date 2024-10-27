
import streamlit as st
import pickle
from surprise import SVD

# Load the SVD model and movie data
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Function to get top movie recommendations for a user
def get_svd_recommendations(user_id, svd_model, movie_ratings, movies, n_recommendations=10):
    # Get movies rated by user
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    # Filter out movies that user hasn't rated
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    
    # Predict ratings for unrated movies
    pred_ratings = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(pred_ratings, key=lambda x: x.est, reverse=True)
    
    # Get top n recommendations
    top_recommendations = sorted_predictions[:n_recommendations]
    
    # Extract movie titles and predicted ratings
    recommendations = [(movies[movies['movieId'] == recommendation.iid]['title'].values[0], recommendation.est) for recommendation in top_recommendations]
    return recommendations

# Streamlit app layout
st.title("Movie Recommendation System with SVD")
st.write("This system provides personalized movie recommendations using an SVD model.")

# Input for User ID
user_id = st.number_input("Enter User ID", min_value=1, max_value=610, value=1, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = get_svd_recommendations(user_id, svd_model, movie_ratings, movies, 10)
    st.write(f"Top 10 movie recommendations for User {user_id}:")
    for movie_title, rating in recommendations:
        st.write(f"{movie_title} (Estimated Rating: {rating:.2f})")
