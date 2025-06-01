from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session and flash messages

# User database (in a real app, use a proper database)
USERS_DB_FILE = 'users.json'

# Check if users.json exists, if not create it
if not os.path.exists(USERS_DB_FILE):
    with open(USERS_DB_FILE, 'w') as f:
        json.dump({}, f)

def load_users():
    with open(USERS_DB_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f)

# Movie Recommender Class
class MovieRecommender:
    def __init__(self, model_dir='model'):
        """Initialize the movie recommender with trained model"""
        self.model_dir = model_dir
        self.load_model()
        
    def load_model(self):
        """Load the trained model and related data"""
        try:
            # Load movie dataframe
            self.movies_df = pd.read_csv('moodflix_movies.csv')  # Make sure this file exists
            
            # Create features column
            self._create_features()
            
            # Initialize and fit TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['Features'])
            
            # Define mood keywords (simplified for this example)
            self.mood_keywords = {
                'happy': ['comedy', 'funny', 'lighthearted', 'happy', 'joyful'],
                'sad': ['drama', 'emotional', 'sad', 'tearjerker', 'heartbreaking'],
                'excited': ['action', 'adventure', 'thriller', 'exciting', 'fast-paced'],
                'scared': ['horror', 'scary', 'frightening', 'terror', 'creepy'],
                'romantic': ['romance', 'love', 'relationship', 'passionate', 'affection']
            }
            
            print(f"Model loaded successfully with {len(self.movies_df)} movies")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _create_features(self):
        """Create features column from movie data"""
        features = []
        for _, row in self.movies_df.iterrows():
            feature_parts = []
            
            # Add title
            feature_parts.append(str(row['Movie Name']).lower())
            
            # Add genre (weighted more heavily)
            if 'Genre' in row:
                feature_parts.append(str(row['Genre']).lower())
                feature_parts.append(str(row['Genre']).lower())  # Duplicate for more weight
            
            # Add actors
            if 'Actors' in row:
                feature_parts.append(str(row['Actors']).lower().replace(',', ' '))
            
            # Add directors
            if 'Directors' in row:
                feature_parts.append(str(row['Directors']).lower().replace(',', ' '))
            
            # Add plot keywords if available
            if 'Plot Keywords' in row:
                feature_parts.append(str(row['Plot Keywords']).lower())
            
            # Combine all parts
            features.append(' '.join(feature_parts))
        
        self.movies_df['Features'] = features
    
    def get_recommendations(self, query, top_n=5):
        """Get movie recommendations based on a user query"""
        try:
            # Convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            
            # Check if query is a recognized mood
            if query_lower in self.mood_keywords:
                # Use mood keywords as the query
                query = ' '.join(self.mood_keywords[query_lower])
            
            # Transform the query using the trained vectorizer
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Compute similarity between query and all movies
            similarity = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get indices of top similar movies
            indices = similarity.argsort()[::-1][:top_n]
            
            # Get the recommended movies
            recommendations = []
            for idx in indices:
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie['Movie Name'],
                    'genre': movie['Genre'] if 'Genre' in movie else 'Unknown',
                    'year': int(movie['Released Year']) if 'Released Year' in movie else 0,
                    'rating': float(movie['Ratings']) if 'Ratings' in movie else 0.0,
                    'actors': movie['Actors'] if 'Actors' in movie else 'Unknown',
                    'director': movie['Directors'] if 'Directors' in movie else 'Unknown',
                    'similarity_score': float(similarity[idx])
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            return []

    def get_available_moods(self):
        """Get list of available moods"""
        return list(self.mood_keywords.keys())

# Initialize the recommender
recommender = MovieRecommender()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        
        # Check if email already exists
        for user_id, user_data in users.items():
            if user_data.get('email') == email:
                flash('Email already registered')
                return redirect(url_for('register'))
        
        # Create new user
        user_id = str(uuid.uuid4())
        users[user_id] = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'favorites': []
        }
        
        save_users(users)
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        
        # Find user by email
        for user_id, user_data in users.items():
            if user_data.get('email') == email:
                if check_password_hash(user_data.get('password'), password):
                    session['user_id'] = user_id
                    session['username'] = user_data.get('username')
                    flash('Login successful!')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid password')
                    return redirect(url_for('login'))
        
        flash('Email not found')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get available moods
    moods = recommender.get_available_moods()
    
    return render_template('dashboard.html', username=session.get('username'), moods=moods)

@app.route('/search', methods=['POST'])
def search():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    query = request.form.get('query')
    recommendations = recommender.get_recommendations(query, top_n=5)
    
    return jsonify(recommendations)

@app.route('/mood', methods=['POST'])
def mood_search():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    mood = request.form.get('mood')
    recommendations = recommender.get_recommendations(mood, top_n=5)
    
    return jsonify(recommendations)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out')
    return redirect(url_for('index'))

@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    movie_title = request.form.get('movie_title')
    
    users = load_users()
    user_id = session['user_id']
    
    if movie_title not in users[user_id]['favorites']:
        users[user_id]['favorites'].append(movie_title)
        save_users(users)
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Movie already in favorites'})

@app.route('/get_favorites')
def get_favorites():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    users = load_users()
    user_id = session['user_id']
    favorites = users[user_id]['favorites']
    
    return jsonify(favorites)

if __name__ == '__main__':
    app.run(debug=True)