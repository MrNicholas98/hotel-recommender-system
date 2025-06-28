import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Sample Hotel Data ---
hotel_data = pd.DataFrame({
    'hotel_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': [
        'Grand Royal Hotel', 'Ocean View Resort', 'City Center Inn',
        'Quiet Lakeside Lodge', 'Luxury Beachfront', 'Budget Stay',
        'Family Fun Hotel', 'Downtown Suites', 'Mountain Retreat',
        'Cozy Guesthouse'
    ],
    'location': [
        'Downtown', 'Beach', 'Downtown', 'Nature', 'Beach',
        'City', 'Resort Area', 'Downtown', 'Nature', 'Rural'
    ],
    'price_range': [
        '$$$', '$$$$', '$$', '$', '$$$$$',
        '$', '$$$', '$$$', '$$', '$$'
    ],
    'amenities': [
        'Pool, Gym, Spa', 'Beach Access, Pool', 'Breakfast, WiFi',
        'Lake View, Hiking', 'Private Beach, Spa, Pool', 'WiFi, Parking',
        'Kids Club, Pool', 'Gym, Business Center', 'Hiking, Views',
        'Garden, Quiet'
    ],
    'star_rating': [5, 4, 3, 4, 5, 2, 4, 3, 3, 2],
    'description': [
        'Elegant hotel in the heart of the city with premium services.',
        'Stunning ocean views and direct beach access, perfect for relaxation.',
        'Affordable and convenient lodging in the city center for business travelers.',
        'Peaceful retreat by a serene lake, ideal for nature lovers.',
        'Ultimate luxury experience with exclusive beach access and top-tier amenities.',
        'No-frills accommodation, clean and basic amenities for budget travelers.',
        'Great for families with kids club and large swimming pools.',
        'Modern suites in the bustling downtown area, close to attractions.',
        'Rustic charm amidst beautiful mountains, great for outdoor activities.',
        'Charming and quiet guesthouse in a rural setting, very personal service.'
    ]
})

# --- Sample User Ratings Data ---
user_ratings = pd.DataFrame({
    'user_id': ['UserA', 'UserB', 'UserC', 'UserD', 'UserE'],
    'hotel_1': [5, np.nan, np.nan, 4, 3],
    'hotel_2': [np.nan, 4, 5, np.nan, np.nan],
    'hotel_3': [3, np.nan, np.nan, np.nan, 5],
    'hotel_4': [np.nan, 5, np.nan, 3, np.nan],
    'hotel_5': [np.nan, np.nan, 4, np.nan, 4],
    'hotel_6': [2, np.nan, np.nan, np.nan, np.nan],
    'hotel_7': [np.nan, np.nan, 3, 5, np.nan],
    'hotel_8': [4, np.nan, np.nan, np.nan, 2],
    'hotel_9': [np.nan, 3, np.nan, np.nan, np.nan],
    'hotel_10': [np.nan, np.nan, 2, np.nan, 3]
}).set_index('user_id')

# --- CONTENT-BASED FILTERING (CBF) IMPLEMENTATION (from before) ---

# Feature Preprocessing: Combine relevant features into a single string
hotel_data['combined_features'] = hotel_data['location'] + ' ' + \
                                  hotel_data['price_range'] + ' ' + \
                                  hotel_data['amenities'] + ' ' + \
                                  hotel_data['star_rating'].astype(str) + ' ' + \
                                  hotel_data['description']

# Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(hotel_data['combined_features'])

# Calculate Cosine Similarity between Hotels
hotel_similarity_matrix = cosine_similarity(tfidf_matrix)

# Function to get content-based recommendations
def get_content_based_recommendations(hotel_id, top_n=5):
    """
    Generates content-based recommendations for a given hotel.
    """
    idx = hotel_data[hotel_data['hotel_id'] == hotel_id].index[0]
    sim_scores = list(enumerate(hotel_similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    hotel_indices = [i[0] for i in sim_scores]
    return hotel_data.iloc[hotel_indices]

# --- COLLABORATIVE FILTERING (CF) IMPLEMENTATION ---

# Fill NaN values with 0 for similarity calculation (assuming 0 means no rating)
# This is important because cosine_similarity does not handle NaN directly.
# We are filling with 0, as absence of a rating doesn't necessarily mean a low rating,
# but for similarity calculation, it implies no agreement/disagreement.
user_ratings_filled = user_ratings.fillna(0)

# Calculate User Similarity (using cosine similarity on user ratings)
# This matrix tells us how similar each user is to every other user based on their rating patterns.
# user_similarity_matrix[i][j] is the similarity between user i and user j.
user_similarity_matrix = cosine_similarity(user_ratings_filled)

# Convert the user similarity matrix into a DataFrame for easier indexing
user_similarity_df = pd.DataFrame(user_similarity_matrix,
                                  index=user_ratings.index,
                                  columns=user_ratings.index)

# Function to get collaborative-based recommendations
def get_collaborative_based_recommendations(user_id, top_n=5):
    """
    Generates collaborative-based recommendations for a given user.

    Args:
        user_id (str): The ID of the user for whom to generate recommendations.
        top_n (int): The number of top hotels to recommend.

    Returns:
        pandas.DataFrame: A DataFrame of recommended hotels, sorted by predicted rating.
    """
    # Get the index of the current user
    user_idx = user_ratings.index.get_loc(user_id)

    # Get the similarity scores for this user with all other users
    # We get the row corresponding to the current user from the similarity matrix
    similar_users = user_similarity_df[user_id].drop(user_id) # Exclude user itself

    # Get top similar users (e.g., top 3)
    # We only consider users who have some similarity (score > 0)
    # and sort them by similarity in descending order
    top_similar_users = similar_users[similar_users > 0].sort_values(ascending=False)

    # Initialize predicted ratings for unrated hotels
    # We'll store potential recommendations with their predicted scores here
    predicted_ratings = pd.Series(dtype=float)

    # Get hotels the current user has NOT rated (these are potential recommendations)
    user_rated_hotels = user_ratings.loc[user_id][user_ratings.loc[user_id].notna()].index.tolist()
    all_hotels_columns = [col for col in user_ratings.columns if col.startswith('hotel_')]
    unrated_hotels = [hotel for hotel in all_hotels_columns if hotel not in user_rated_hotels]

    # For each unrated hotel, predict the rating
    for hotel_col in unrated_hotels:
        weighted_sum = 0
        similarity_sum = 0

        # Iterate through top similar users
        for s_user_id, similarity_score in top_similar_users.items():
            # Get the rating of the current hotel by the similar user
            s_user_rating = user_ratings.loc[s_user_id, hotel_col]

            # If the similar user has rated this hotel, use their rating for prediction
            if pd.notna(s_user_rating): # Check if rating is not NaN
                weighted_sum += (s_user_rating * similarity_score)
                similarity_sum += similarity_score

        if similarity_sum > 0:
            predicted_ratings[hotel_col] = weighted_sum / similarity_sum
        else:
            predicted_ratings[hotel_col] = 0 # No similar users rated this, so prediction is 0

    # Sort predicted ratings in descending order and get top N
    recommended_hotel_cols = predicted_ratings.sort_values(ascending=False).head(top_n).index.tolist()

    # Convert hotel column names (e.g., 'hotel_1') back to hotel_id (e.g., 1)
    recommended_hotel_ids = [int(col.split('_')[1]) for col in recommended_hotel_cols]

    # Return the actual hotel data for these recommended IDs
    return hotel_data[hotel_data['hotel_id'].isin(recommended_hotel_ids)]


# --- TESTING COLLABORATIVE-BASED RECOMMENDATIONS ---
print("\n--- Collaborative-Based Recommendations Test ---")

# Example: Get recommendations for UserA
# UserA has rated Hotel 1, 3, 6, 8
recommended_hotels_cbf = get_collaborative_based_recommendations(user_id='UserA')
print("\nRecommendations for UserA:")
print(recommended_hotels_cbf[['hotel_id', 'name', 'location', 'star_rating']])

# Example: Get recommendations for UserB
# UserB has rated Hotel 2, 4, 9
recommended_hotels_cbf_2 = get_collaborative_based_recommendations(user_id='UserB')
print("\nRecommendations for UserB:")
print(recommended_hotels_cbf_2[['hotel_id', 'name', 'location', 'star_rating']])