import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify # Import Flask components
from flask_cors import CORS # Import CORS for cross-origin requests

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

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

# --- CONTENT-BASED FILTERING (CBF) SETUP ---

hotel_data['combined_features'] = hotel_data['location'] + ' ' + \
                                  hotel_data['price_range'] + ' ' + \
                                  hotel_data['amenities'] + ' ' + \
                                  hotel_data['star_rating'].astype(str) + ' ' + \
                                  hotel_data['description']

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(hotel_data['combined_features'])
hotel_similarity_matrix = cosine_similarity(tfidf_matrix)

def get_content_based_recommendations(hotel_id, top_n=5):
    idx = hotel_data[hotel_data['hotel_id'] == hotel_id].index[0]
    sim_scores = list(enumerate(hotel_similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    hotel_indices = [i[0] for i in sim_scores]
    return hotel_data.iloc[hotel_indices]

# --- COLLABORATIVE FILTERING (CF) SETUP ---

user_ratings_filled = user_ratings.fillna(0)
user_similarity_matrix = cosine_similarity(user_ratings_filled)
user_similarity_df = pd.DataFrame(user_similarity_matrix,
                                  index=user_ratings.index,
                                  columns=user_ratings.index)

def get_collaborative_based_recommendations(user_id, top_n=5):
    user_idx = user_ratings.index.get_loc(user_id)
    similar_users = user_similarity_df[user_id].drop(user_id)
    top_similar_users = similar_users[similar_users > 0].sort_values(ascending=False)

    predicted_ratings = pd.Series(dtype=float)
    user_rated_hotels = user_ratings.loc[user_id][user_ratings.loc[user_id].notna()].index.tolist()
    all_hotels_columns = [col for col in user_ratings.columns if col.startswith('hotel_')]
    unrated_hotels = [hotel for hotel in all_hotels_columns if hotel not in user_rated_hotels]

    for hotel_col in unrated_hotels:
        weighted_sum = 0
        similarity_sum = 0
        for s_user_id, similarity_score in top_similar_users.items():
            s_user_rating = user_ratings.loc[s_user_id, hotel_col]
            if pd.notna(s_user_rating):
                weighted_sum += (s_user_rating * similarity_score)
                similarity_sum += similarity_score

        if similarity_sum > 0:
            predicted_ratings[hotel_col] = weighted_sum / similarity_sum
        else:
            predicted_ratings[hotel_col] = 0

    recommended_hotel_cols = predicted_ratings.sort_values(ascending=False).head(top_n).index.tolist()
    recommended_hotel_ids = [int(col.split('_')[1]) for col in recommended_hotel_cols]
    return hotel_data[hotel_data['hotel_id'].isin(recommended_hotel_ids)]

# --- HYBRID RECOMMENDATION SYSTEM IMPLEMENTATION ---

def get_hybrid_recommendations(user_id, top_n=5, cbf_weight=0.5, cf_weight=0.5):
    if not np.isclose(cbf_weight + cf_weight, 1.0):
        print("Warning: CBF and CF weights do not sum to 1. Normalizing...")
        total_weight = cbf_weight + cf_weight
        cbf_weight /= total_weight
        cf_weight /= total_weight

    user_rated_hotels = user_ratings.loc[user_id][user_ratings.loc[user_id].notna()]
    rated_hotel_ids = [int(col.split('_')[1]) for col in user_rated_hotels.index.tolist()]
    all_hotel_ids = hotel_data['hotel_id'].tolist()
    hybrid_scores = pd.Series(0.0, index=all_hotel_ids)

    cbf_candidate_scores = {}
    for rated_hotel_col, rating in user_rated_hotels.items():
        rated_hotel_id = int(rated_hotel_col.split('_')[1])
        idx = hotel_data[hotel_data['hotel_id'] == rated_hotel_id].index[0]
        current_hotel_sims = pd.Series(hotel_similarity_matrix[idx], index=hotel_data['hotel_id'])
        weighted_sims = current_hotel_sims * rating

        for hotel_id, score in weighted_sims.items():
            if hotel_id not in rated_hotel_ids:
                cbf_candidate_scores[hotel_id] = max(cbf_candidate_scores.get(hotel_id, 0), score)

    if cbf_candidate_scores:
        max_cbf_score = max(cbf_candidate_scores.values())
        if max_cbf_score > 0:
            for hotel_id in cbf_candidate_scores:
                cbf_candidate_scores[hotel_id] /= max_cbf_score

    for hotel_id, score in cbf_candidate_scores.items():
        hybrid_scores[hotel_id] += cbf_weight * score

    cf_predicted_ratings = pd.Series(dtype=float)
    user_idx = user_ratings.index.get_loc(user_id)
    similar_users = user_similarity_df[user_id].drop(user_id)
    top_similar_users = similar_users[similar_users > 0].sort_values(ascending=False)
    unrated_hotels_cols = [col for col in user_ratings.columns if col.startswith('hotel_') and int(col.split('_')[1]) not in rated_hotel_ids]

    for hotel_col in unrated_hotels_cols:
        weighted_sum = 0
        similarity_sum = 0
        for s_user_id, similarity_score in top_similar_users.items():
            s_user_rating = user_ratings.loc[s_user_id, hotel_col]
            if pd.notna(s_user_rating):
                weighted_sum += (s_user_rating * similarity_score)
                similarity_sum += similarity_score

        if similarity_sum > 0:
            cf_predicted_ratings[hotel_col] = weighted_sum / similarity_sum
        else:
            cf_predicted_ratings[hotel_col] = 0

    if cf_predicted_ratings.max() > 0:
        cf_predicted_ratings = cf_predicted_ratings / cf_predicted_ratings.max()

    for hotel_col, score in cf_predicted_ratings.items():
        hotel_id = int(hotel_col.split('_')[1])
        hybrid_scores[hotel_id] += cf_weight * score

    hybrid_scores = hybrid_scores.drop(rated_hotel_ids, errors='ignore')
    recommended_hotel_ids = hybrid_scores.sort_values(ascending=False).head(top_n).index.tolist()
    return hotel_data[hotel_data['hotel_id'].isin(recommended_hotel_ids)]

# --- FLASK API ENDPOINT ---

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to get hybrid hotel recommendations for a given user.
    Expects a JSON payload with 'user_id' and optional 'top_n'.
    """
    data = request.get_json()
    user_id = data.get('user_id')
    top_n = data.get('top_n', 5) # Default to 5 recommendations

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    # Check if user_id exists in our sample data
    if user_id not in user_ratings.index:
        return jsonify({"error": f"User ID '{user_id}' not found in sample data. Available users: {list(user_ratings.index)}"}), 404

    try:
        # Get recommendations using our hybrid model
        recommendations_df = get_hybrid_recommendations(user_id=user_id, top_n=top_n)

        # Convert DataFrame to a list of dictionaries for JSON response
        recommendations_list = recommendations_df.to_dict(orient='records')

        return jsonify(recommendations_list)

    except Exception as e:
        # Basic error handling for unexpected issues
        return jsonify({"error": str(e)}), 500

# --- Main execution block for running the Flask app ---
if __name__ == '__main__':
    # To run the Flask app, use 'flask run' command in terminal
    # For development, you can run it directly like this, but 'flask run' is preferred
    # app.run(debug=True, port=5000) # debug=True enables auto-reloading and better error messages
    print("\n--- Flask Server Ready ---")
    print("To run the Flask server, open a NEW terminal in this folder and type:")
    print("flask --app main run --debug --port 5000")
    print("Then, access the web interface or send POST requests to http://127.0.0.1:5000/recommend")