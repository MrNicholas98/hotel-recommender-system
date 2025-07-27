Hotel Recommendation System
Project Overview
This project implements a Hotel Recommendation System designed to provide personalized hotel suggestions to users. It leverages a Hybrid Recommendation Model, combining the strengths of both Content-Based Filtering and Collaborative Filtering to deliver accurate and diverse recommendations. The system features an interactive web interface, allowing users to easily select a profile and instantly view tailored hotel suggestions.

Features
Hybrid Recommendation Model: Combines two powerful recommendation techniques:

Content-Based Filtering (CBF): Recommends hotels based on their intrinsic features (e.g., location, amenities, price range, star rating, description). If a user likes a hotel with specific characteristics, the system suggests other hotels with similar characteristics.

Collaborative Filtering (CF): Recommends hotels by identifying users with similar historical rating patterns. If users A and B have similar tastes, and user A liked a hotel that user B hasn't seen, it recommends that hotel to user B.

Interactive Web Interface: A user-friendly HTML/CSS/JavaScript frontend allows for easy interaction.

Python Flask API Backend: A lightweight Flask web server powers the recommendation logic, serving personalized hotel lists to the frontend.

Dynamic Updates: Recommendations update instantly on the web page without requiring a full page refresh.

Scalable Architecture: The modular design allows for future expansion with more complex data and features.

Technologies Used
Python: The core programming language for the recommendation engine.

Pandas: For efficient data manipulation and analysis of hotel and user rating datasets.

NumPy: Essential for numerical operations, especially within scikit-learn.

Scikit-learn: A powerful machine learning library used for:

TfidfVectorizer: To convert text features (amenities, descriptions) into numerical vectors for CBF.

cosine_similarity: To calculate similarity between hotels (for CBF) and between users (for CF).

Flask: A micro web framework for Python, used to create the RESTful API backend.

Flask-Cors: A Flask extension to handle Cross-Origin Resource Sharing, enabling seamless communication between the web frontend and the Flask API.

HTML: For structuring the web page.

CSS (Tailwind CSS): For modern, responsive, and aesthetically pleasing styling of the web interface.

JavaScript: For handling frontend interactions, making API calls to the Flask backend, and dynamically displaying recommendations.

How It Works
Data Loading: Sample hotel feature data and user rating data are loaded into Pandas DataFrames.

Content-Based Processing: Hotel features are combined and vectorized using TF-IDF. Cosine similarity is then calculated between all hotels to determine feature-based resemblance.

Collaborative Filtering Processing: User rating data is used to calculate similarity between users based on their shared preferences.

Hybrid Recommendation Logic: The get_hybrid_recommendations function combines scores from both CBF and CF. For a given user, it considers hotels they've liked (CBF) and hotels liked by similar users (CF) to generate a weighted recommendation score.

Flask API: A /recommend endpoint in the Flask application receives a user_id from the frontend. It calls the hybrid recommendation function and returns the recommended hotels as a JSON response.

Frontend Interaction: The index.html page allows users to select a user_id. Upon clicking "Get Recommendations", JavaScript sends this ID to the Flask API. The received recommendations are then dynamically rendered on the page.

Setup and Running the Project
To run this project locally, follow these steps:

Clone the Repository:

git clone https://github.com/MrNicholas98/hotel-recommender-system.git
cd hotel-recommender-system

Create and Activate a Virtual Environment:

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

Install Dependencies:

pip install pandas scikit-learn Flask Flask-Cors numpy

Start the Flask Backend Server:
Open a new terminal in your hotel-recommender-system directory (ensure (venv) is active in this new terminal too).

flask --app main run --debug --port 5000

Leave this terminal running. It will continuously serve your recommendations.

Open the Web Interface:

In VS Code's Explorer, right-click on index.html.

Select "Open with Live Server" (requires the Live Server VS Code extension).

Your web browser will open, displaying the interactive recommendation system.

Interact with the System:

Select a user from the dropdown.

Click "Get Recommendations" to see personalized hotel suggestions.

Future Enhancements
Implement a database (e.g., SQLite, PostgreSQL) for persistent storage of hotel data and user ratings.

Allow users to input new ratings or add new hotels.

Add user authentication.

Improve the web interface with more advanced filtering and display options.

Integrate more sophisticated recommendation algorithms.

Developed by [Nicholas Chisom Nneke/MrNicholas98]
