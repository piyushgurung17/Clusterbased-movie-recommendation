import streamlit as st
import pandas as pd
import pickle

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pc.pkl", "rb") as f:
    pca = pickle.load(f)

# Load KMeans model
with open("kmeansmovie.pkl", "rb") as f:
    model = pickle.load(f)

# Load recomm dictionary
with open("recomm.pkl", "rb") as f:
    recomm = pickle.load(f)

# Streamlit UI
st.title("Cluster-Based Movie Recommendation System")
st.image('https://miro.medium.com/v2/resize:fit:1400/1*xCa6ShdprzAMRh7qSCEM9A.png')
# Genre encoding
genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-fi", "Thriller"])
genre_mapping = {
    "Action": 0,
    "Comedy": 1,
    "Drama": 2,
    "Horror": 3,
    "Romance": 4,
    "Sci-fi": 5,
    "Thriller": 6
}
gen = genre_mapping[genre]

# Other inputs
rating = st.slider("Rating", 1.0, 5.0)
time_spent = st.number_input("Time Spent (minutes)")
location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
location_mapping = {
    "Rural": 0,
    "Suburban": 1,
    "Urban": 2
}
loc = location_mapping[location]

user = st.text_input("User ID")

# Form input dataframe
input_df = pd.DataFrame({
    'genre': [gen],
    'rating': [rating],
    'time_spent_minutes': [time_spent],
    'location': [loc],
    'user': [user]
})

input_features = input_df.drop(columns=["user"])  # Drop user column if it was not used in PCA

# Scale and apply PCA
input_scaled = scaler.transform(input_features)
input_pca = pca.transform(input_scaled)

movie_df = pd.read_csv("movieinfo.csv")  # Make sure this file exists in your project folder

# Create dictionary for fast lookup
movie_dict = dict(zip(movie_df['movie_id'], movie_df['movie_name']))

# Predict cluster and recommend
if st.button("Get Recommendations"):
    cluster = model.predict(input_pca)[0]
    st.subheader(f"📊 Predicted Cluster: {cluster}")
    
    try:
        recommended_movies = recomm[cluster][gen]['movie']
        genre_users = recomm[cluster][gen]['user']
        
        if user in genre_users:
            st.success("🎯 Based on your cluster and genre preference, here are your top movie recommendations:")
        else:
            st.info("ℹ You haven't interacted with this genre much, but here are top-rated movies in your cluster and genre:")

        # for movie in recommended_movies:
        #     st.write(f"🎬 Movie ID: {movie}")
        for movie in recommended_movies:
            movie_name = movie_dict.get(movie, "Unknown Title")
            st.write(f"🎬 Movie ID: {movie} — **{movie_name}**")
    except:
        st.error("No recommendations found for this combination.")
