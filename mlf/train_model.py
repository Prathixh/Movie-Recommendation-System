import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class MovieRecommenderModel:
    def __init__(self):
        """Initialize the movie recommender model"""
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.cosine_sim = None
        
        # Define mood to keyword mappings
        self.mood_keywords = {
            'happy': ['comedy', 'uplifting', 'cheerful', 'heartwarming', 'inspiring', 'whimsical'],
            'sad': ['drama', 'emotional', 'touching', 'heartbreaking', 'melancholy'],
            'excited': ['action', 'adventure', 'thriller', 'suspense', 'adrenaline-fueled', 'exciting'],
            'scared': ['horror', 'thriller', 'suspense', 'dark', 'frightening'],
            'romantic': ['romance', 'love', 'relationship', 'passion', 'emotional', 'dreamy'],
            'thoughtful': ['documentary', 'philosophical', 'thought-provoking', 'contemplative'],
            'intense': ['drama', 'thriller', 'crime', 'intense', 'gritty', 'suspenseful', 'tense'],
            'relaxed': ['animation', 'family', 'chill', 'gentle', 'feel-good'],
            'inspired': ['biography', 'inspiring', 'hopeful', 'uplifting'],
            'epic': ['adventure', 'action', 'epic', 'fantasy'],
            'amused': ['comedy', 'quirky', 'whimsical', 'darkly comedic'],
            'intrigued': ['mystery', 'intriguing', 'mind-bending', 'thought-provoking']
        }
        
    def load_data(self, csv_file='moodflix_movies.csv'):
        """
        Load movie data from CSV file
        
        Parameters:
        csv_file (str): Path to the CSV file with movie data (default: moodflix_movies.csv)
        """
        self.movies_df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.movies_df)} movies from {csv_file}")
        return self
    
    def prepare_data(self):
        """Prepare data for recommendations by creating feature text"""
        self.movies_df['Features'] = ''
        
        self.movies_df['Features'] += self.movies_df['Genre'].apply(lambda x: x.lower() + ' ' + x.lower() + ' ')
        self.movies_df['Features'] += self.movies_df['Actors'].apply(lambda x: x.lower().replace(',', ' ') + ' ')
        self.movies_df['Features'] += self.movies_df['Co-Actors'].apply(lambda x: x.lower().replace(',', ' ') + ' ')
        self.movies_df['Features'] += self.movies_df['Directors'].apply(lambda x: x.lower().replace(',', ' ') + ' ')
        self.movies_df['Features'] += self.movies_df['Mood Type'].apply(lambda x: x.lower() + ' ' + x.lower() + ' ')
        self.movies_df['Features'] += self.movies_df['Movie Name'].apply(lambda x: x.lower())
        self.movies_df['Features'] = self.movies_df['Features'].apply(lambda x: x.replace('-', ' '))
        
        print("Features created for all movies")
        return self
    
    def train_model(self):
        """Train the TF-IDF model for content-based recommendations"""
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['Features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("Model trained successfully")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self
    
    def save_model(self, model_dir='model'):
        """Save the trained model and related data"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        with open(f"{model_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        np.save(f"{model_dir}/cosine_sim.npy", self.cosine_sim)
        self.movies_df.to_pickle(f"{model_dir}/movies_df.pkl")
        
        with open(f"{model_dir}/mood_keywords.pkl", 'wb') as f:
            pickle.dump(self.mood_keywords, f)
        
        print(f"Model saved to {model_dir} directory")
        return self
    
    def get_recommendations(self, query, top_n=5):
        """Get movie recommendations based on a query"""
        query_lower = query.lower()
        if query_lower in self.mood_keywords:
            query = ' '.join(self.mood_keywords[query_lower])
        
        query_vec = self.tfidf_vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        indices = similarity.argsort()[::-1][:top_n]
        
        recommendations = []
        for idx in indices:
            movie = self.movies_df.iloc[idx]
            recommendations.append({
                'title': movie['Movie Name'],
                'genre': movie['Genre'],
                'year': movie['Released Year'],
                'rating': movie['Ratings'],
                'actors': f"{movie['Actors']}, {movie['Co-Actors']}",
                'director': movie['Directors'],
                'mood': movie['Mood Type'],
                'similarity_score': similarity[idx]
            })
        
        return recommendations
    
    def evaluate_model(self):
        """Evaluate the model using a sample of mood queries"""
        evaluation_results = {}
        sample_moods = list(self.mood_keywords.keys())[:5]
        
        for mood in sample_moods:
            recommendations = self.get_recommendations(mood, top_n=3)
            evaluation_results[mood] = [rec['title'] for rec in recommendations]
            
            print(f"\nRecommendations for mood '{mood}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']} ({rec['genre']}, {rec['year']}) - {rec['mood']}")
        
        return evaluation_results

def main():
    print("Starting model training process...")
    model = MovieRecommenderModel()
    model.load_data('moodflix_movies.csv')
    model.prepare_data()
    model.train_model()
    
    print("\nEvaluating model performance:")
    model.evaluate_model()
    
    model.save_model()
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
