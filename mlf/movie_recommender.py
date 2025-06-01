import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommender:
    def __init__(self, model_dir='model'):
        """
        Initialize the movie recommender with trained model
        
        Parameters:
        model_dir (str): Directory containing the model files
        """
        self.model_dir = model_dir
        self.load_model()
        
    def load_model(self):
        """Load the trained model and related data"""
        try:
            # Load vectorizer
            with open(f"{self.model_dir}/tfidf_vectorizer.pkl", 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Load cosine similarity matrix
            self.cosine_sim = np.load(f"{self.model_dir}/cosine_sim.npy")
            
            # Load movie dataframe
            self.movies_df = pd.read_pickle(f"{self.model_dir}/movies_df.pkl")
            
            # Load mood keywords
            with open(f"{self.model_dir}/mood_keywords.pkl", 'rb') as f:
                self.mood_keywords = pickle.load(f)
                
            print(f"Model loaded successfully with {len(self.movies_df)} movies")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # If model not found, show instructions
            print("Please run train_model.py first to create the recommendation model")
            return False
    
    def get_recommendations(self, query, top_n=5):
        """
        Get movie recommendations based on a user query
        
        Parameters:
        query (str): User query (mood or keywords)
        top_n (int): Number of recommendations to return
        
        Returns:
        list: List of recommended movie dictionaries
        """
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Check if query is a recognized mood
        if query_lower in self.mood_keywords:
            print(f"Recognized mood: {query_lower}")
            # Use mood keywords as the query
            query = ' '.join(self.mood_keywords[query_lower])
            print(f"Using mood keywords: {query}")
        
        # Transform the query using the trained vectorizer
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Check if Features column exists
        if 'Features' in self.movies_df.columns:
            # Use existing Features column
            self.tfidf_matrix = self.tfidf_vectorizer.transform(self.movies_df['Features'])
        else:
            # Recreate Features column if it doesn't exist
            print("Features column not found. Regenerating features...")
            self.movies_df['Features'] = ''
            
            # Add genre (weight: high)
            if 'Genre' in self.movies_df.columns:
                self.movies_df['Features'] += self.movies_df['Genre'].apply(lambda x: str(x).lower() + ' ' + str(x).lower() + ' ')
            
            # Add actors
            if 'Actors' in self.movies_df.columns:
                self.movies_df['Features'] += self.movies_df['Actors'].apply(lambda x: str(x).lower().replace(',', ' ') + ' ')
            
            # Add co-actors
            if 'Co-Actors' in self.movies_df.columns:
                self.movies_df['Features'] += self.movies_df['Co-Actors'].apply(lambda x: str(x).lower().replace(',', ' ') + ' ')
            
            # Add directors
            if 'Directors' in self.movies_df.columns:
                self.movies_df['Features'] += self.movies_df['Directors'].apply(lambda x: str(x).lower().replace(',', ' ') + ' ')
            
            # Add mood
            if 'Mood Type' in self.movies_df.columns:
                self.movies_df['Features'] += self.movies_df['Mood Type'].apply(lambda x: str(x).lower() + ' ' + str(x).lower() + ' ')
            
            # Add title
            self.movies_df['Features'] += self.movies_df['Movie Name'].apply(lambda x: str(x).lower())
            
            # Clean the features text
            self.movies_df['Features'] = self.movies_df['Features'].apply(lambda x: str(x).replace('-', ' '))
            
            # Transform the regenerated features
            self.tfidf_matrix = self.tfidf_vectorizer.transform(self.movies_df['Features'])
        
        # Compute similarity between query and all movies
        similarity = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get indices of top similar movies
        indices = similarity.argsort()[::-1][:top_n]
        
        # Get the recommended movies
        recommendations = []
        for idx in indices:
            movie = self.movies_df.iloc[idx]
            if similarity[idx] > 0:  # Only include if there's some similarity
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
    
    def print_recommendations(self, recommendations):
        """
        Print recommendations in a formatted way
        
        Parameters:
        recommendations (list): List of movie recommendation dictionaries
        """
        if not recommendations:
            print("No recommendations found for your query.")
            return
        
        print("\n===== Movie Recommendations =====")
        for i, movie in enumerate(recommendations, 1):
            print(f"\n{i}. {movie['title']} ({movie['year']}) - {movie['rating']}/10")
            print(f"   Genre: {movie['genre']}")
            print(f"   Mood: {movie['mood']}")
            print(f"   Director: {movie['director']}")
            print(f"   Cast: {movie['actors']}")
            print(f"   Match Score: {movie['similarity_score']:.2f}")
        print("\n================================")

# Main function to run the recommender
def main():
    # Create the recommender
    recommender = MovieRecommender()
    
    # Check if model loaded successfully
    if not hasattr(recommender, 'movies_df'):
        print("Model loading failed. Exiting.")
        return
    
    print("\nMovie Recommendation System")
    print("---------------------------")
    print("You can enter a mood (like 'happy', 'sad', 'excited') or keywords")
    print("Available moods:", ", ".join(recommender.mood_keywords.keys()))
    
    while True:
        # Get user input
        query = input("\nEnter your mood or keywords (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            print("Thank you for using the Movie Recommender!")
            break
        
        # Get recommendations
        recommendations = recommender.get_recommendations(query)
        
        # Display recommendations
        recommender.print_recommendations(recommendations)

if __name__ == "__main__":
    main()
