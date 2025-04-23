import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from joblib import Parallel, delayed
import warnings
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Global variables for preprocessing
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text_batch(texts):
    """Process a batch of texts in parallel"""
    processed = []
    for text in texts:
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-z\s]', '', str(text).lower())
        
        # Split into words
        words = text.split()
        
        # Remove stopwords, lemmatize, and filter short words
        words = [LEMMATIZER.lemmatize(word) for word in words 
                if word not in STOP_WORDS and len(word) > 2]
        
        processed.append(' '.join(words))
    return processed

class MovieGenreClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.genres = None
        self.train_data = None
        
    def load_data(self, train_path, test_path, solution_path):
        """Load and preprocess the dataset"""
        try:
            print(f"Attempting to load data from:")
            print(f"Train path: {train_path}")
            print(f"Test path: {test_path}")
            print(f"Solution path: {solution_path}")
            
            # Verify file existence
            for path in [train_path, test_path, solution_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
            
            # Read the data with correct separator and error handling
            try:
                train_data = pd.read_csv(train_path, sep=' ::: ', header=None, 
                                       names=['id', 'title', 'genre', 'description'], 
                                       engine='python', encoding='utf-8')
                print(f"Successfully loaded train data with {len(train_data)} rows")
                
                test_data = pd.read_csv(test_path, sep=' ::: ', header=None,
                                      names=['id', 'title', 'description'], 
                                      engine='python', encoding='utf-8')
                print(f"Successfully loaded test data with {len(test_data)} rows")
                
                solution_data = pd.read_csv(solution_path, sep=' ::: ', header=None,
                                          names=['id', 'title', 'genre', 'description'], 
                                          engine='python', encoding='utf-8')
                print(f"Successfully loaded solution data with {len(solution_data)} rows")
                
            except pd.errors.EmptyDataError:
                print("Error: One or more data files are empty")
                raise
            except pd.errors.ParserError as e:
                print(f"Error parsing CSV data: {str(e)}")
                print("This might be due to incorrect separator or file format")
                raise
            
            # Store train data for movie prediction
            self.train_data = train_data
            
            # Basic validation
            if len(train_data) == 0 or len(test_data) == 0 or len(solution_data) == 0:
                raise ValueError("One or more datasets are empty")
            
            print("Data loading completed successfully")
            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            print(f"Solution data shape: {solution_data.shape}")
            
            return train_data, test_data, solution_data
            
        except Exception as e:
            print(f"Error in load_data: {str(e)}")
            print("Current working directory:", os.getcwd())
            raise

    def parallel_preprocess(self, texts, batch_size=1000):
        """Preprocess texts in parallel using batches"""
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(tqdm(executor.map(preprocess_text_batch, batches), 
                              total=len(batches), 
                              desc="Processing text"))
        
        # Flatten results
        return [text for batch in results for text in batch]

    def train_model(self, train_data, test_data, solution_data):
        """Train model using mini-batches and parallel processing"""
        import time
        start_time = time.time()
        
        try:
            print("\nStep 1/4: Preprocessing text data...")
            
            # Parallel preprocessing
            print("Processing training data...")
            train_processed = self.parallel_preprocess(train_data['description'].values)
            print("Processing test data...")
            test_processed = self.parallel_preprocess(test_data['description'].values)
            
            print("\nStep 2/4: Converting text to numerical features...")
            tfidf_start = time.time()
            
            # Optimized TF-IDF with better parameters
            self.vectorizer = TfidfVectorizer(
                max_features=15000,  # Increased for better accuracy
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                strip_accents='unicode',
                sublinear_tf=True,
                dtype=np.float32  # Use float32 for memory efficiency
            )
            
            print("Vectorizing training data...")
            X_train = self.vectorizer.fit_transform(train_processed)
            print("Vectorizing test data...")
            X_test = self.vectorizer.transform(test_processed)
            
            y_train = train_data['genre']
            y_test = solution_data['genre']
            
            print(f"Vectorization completed in {(time.time() - tfidf_start):.2f} seconds")
            print(f"Features: {X_train.shape[1]}")
            
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.genres = np.unique(y_train)
            
            print("\nStep 3/4: Training model using mini-batches...")
            
            # Initialize model with optimized parameters
            model = LogisticRegression(
                C=10,
                solver='saga',
                class_weight='balanced',
                max_iter=1000,
                n_jobs=-1,
                tol=0.01,
                verbose=1,
                warm_start=True  # Enable warm start for mini-batch training
            )
            
            # Mini-batch training
            batch_size = 5000
            n_batches = int(np.ceil(X_train.shape[0] / batch_size))
            
            train_start = time.time()
            for epoch in range(2):  # Two epochs for better convergence
                print(f"\nEpoch {epoch + 1}/2")
                for i in tqdm(range(n_batches), desc="Training batches"):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, X_train.shape[0])
                    
                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]
                    
                    model.fit(X_batch, y_batch)
            
            train_time = time.time() - train_start
            
            # Evaluate model
            print("\nStep 4/4: Evaluating model...")
            
            # Use mini-batches for prediction to save memory
            train_preds = []
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                train_preds.extend(model.predict(X_batch))
            
            test_preds = []
            for i in range(0, X_test.shape[0], batch_size):
                X_batch = X_test[i:i + batch_size]
                test_preds.extend(model.predict(X_batch))
            
            train_score = accuracy_score(y_train, train_preds)
            test_score = accuracy_score(y_test, test_preds)
            
            print("\nTraining Summary:")
            print(f"Total time: {(time.time() - start_time):.2f} seconds")
            print(f"Training time: {train_time:.2f} seconds")
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Test accuracy: {test_score:.4f}")
            
            self.model = model
            self.train_data = train_data
            return model, test_score
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def get_feature_importance(self, top_n=20):
        """Get feature importance for the model"""
        if self.model is None or self.feature_names is None:
            return [], []
            
        try:
            if isinstance(self.model, RandomForestClassifier):
                importances = self.model.feature_importances_
            else:
                importances = np.abs(self.model.coef_).mean(axis=0)
            
            # Get top features
            top_indices = importances.argsort()[-top_n:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            return top_features, top_importances
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return [], []

    def predict(self, text):
        """Make predictions with optimized processing"""
        if self.model is None or self.vectorizer is None:
            return "Model not trained", []
            
        try:
            # Preprocess text
            processed_text = preprocess_text_batch([text])[0]
            text_vector = self.vectorizer.transform([processed_text])
            
            # Get prediction and probabilities
            prediction = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]
            
            return prediction, probabilities
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Prediction error", []

    def find_similar_movies(self, text, top_n=5):
        """Find similar movies based on plot description"""
        if self.train_data is None or self.vectorizer is None:
            return []
            
        try:
            # Process input text with more thorough preprocessing
            processed_text = preprocess_text_batch([text])[0]
            text_vector = self.vectorizer.transform([processed_text])
            
            # Process training data in batches
            batch_size = 1000
            similarities = []
            
            for i in range(0, len(self.train_data), batch_size):
                batch = self.train_data.iloc[i:i + batch_size]
                # Process descriptions in the batch
                batch_processed = preprocess_text_batch(batch['description'].values)
                batch_vectors = self.vectorizer.transform(batch_processed)
                
                # Calculate cosine similarity for the batch
                # Normalize vectors for cosine similarity
                text_norm = np.linalg.norm(text_vector.toarray())
                batch_norms = np.linalg.norm(batch_vectors.toarray(), axis=1)
                
                # Calculate cosine similarity
                batch_similarities = np.dot(text_vector.toarray(), batch_vectors.T.toarray())[0]
                batch_similarities = batch_similarities / (text_norm * batch_norms)
                
                # Add results with similarity scores
                for j, sim in enumerate(batch_similarities):
                    idx = i + j
                    # Calculate exact match score using processed text
                    batch_text = batch_processed[j]
                    exact_match = 1.0 if batch_text == processed_text else 0.0
                    # Combine exact match with similarity score
                    combined_score = 0.7 * exact_match + 0.3 * sim
                    
                    similarities.append((
                        self.train_data.iloc[idx]['title'],
                        self.train_data.iloc[idx]['genre'],
                        float(combined_score)
                    ))
            
            # Sort by combined score
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Return top N results, ensuring we have at least some results
            if not similarities:
                return []
            
            # If we have exact matches, return them first
            exact_matches = [s for s in similarities if s[2] >= 0.7]
            if exact_matches:
                return exact_matches[:top_n]
            
            # Otherwise return the most similar movies
            return similarities[:top_n]
            
        except Exception as e:
            print(f"Error finding similar movies: {str(e)}")
            return []

class MovieGenreGUI:
    def __init__(self, classifier):
        self.classifier = classifier
        self.root = tk.Tk()
        self.root.title("Movie Genre Classifier")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Set style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 20, 'bold'))
        self.style.configure('Result.TLabel', font=('Arial', 12))
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Label(main_container, 
                          text="Movie Genre Classifier", 
                          style='Header.TLabel')
        header.pack(pady=(0, 20))
        
        # Left panel for input and results
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel for visualizations
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Input Section
        input_frame = ttk.LabelFrame(left_panel, text="Movie Plot", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Text area for plot input
        self.plot_text = scrolledtext.ScrolledText(
            input_frame, 
            height=6, 
            width=50, 
            font=('Arial', 11),
            wrap=tk.WORD
        )
        self.plot_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Predict button
        predict_button = ttk.Button(
            input_frame, 
            text="Predict Genre", 
            command=self.predict_genre,
            style='TButton'
        )
        predict_button.pack(pady=(0, 5))
        
        # Results Section
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prediction result
        self.prediction_label = ttk.Label(
            results_frame, 
            text="Predicted Genre: ", 
            style='Result.TLabel'
        )
        self.prediction_label.pack(fill=tk.X, pady=5)
        
        # Top genres
        ttk.Label(results_frame, text="Top 3 Genres:", style='Result.TLabel').pack(pady=(10, 5))
        self.top_genres_text = scrolledtext.ScrolledText(
            results_frame, 
            height=4, 
            font=('Arial', 11),
            wrap=tk.WORD
        )
        self.top_genres_text.pack(fill=tk.X, pady=5)
        
        # Similar movies
        ttk.Label(results_frame, text="Similar Movies:", style='Result.TLabel').pack(pady=(10, 5))
        self.similar_movies_text = scrolledtext.ScrolledText(
            results_frame, 
            height=6, 
            font=('Arial', 11),
            wrap=tk.WORD
        )
        self.similar_movies_text.pack(fill=tk.X, pady=5)
        
        # Visualization Section
        viz_frame = ttk.LabelFrame(right_panel, text="Analysis", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Initialize visualization frames
        self.prob_frame = ttk.Frame(self.notebook)
        self.feature_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.prob_frame, text="Genre Probabilities")
        self.notebook.add(self.feature_frame, text="Important Features")
        
        # Initialize canvas holders
        self.prob_canvas = None
        self.feature_canvas = None
    
    def predict_genre(self):
        # Clear previous results
        self.clear_results()
        
        # Get input text
        text = self.plot_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter a movie plot description")
            return
        
        try:
            # Get prediction
            prediction, probabilities = self.classifier.predict(text)
            
            if isinstance(prediction, str) and prediction.startswith("Error"):
                messagebox.showerror("Error", prediction)
                return
            
            # Update prediction label
            self.prediction_label.config(
                text=f"Predicted Genre: {prediction}",
                foreground='#2E7D32'
            )
            
            # Update top genres
            prob_df = pd.DataFrame({
                'Genre': self.classifier.genres,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            top_3_text = "Top 3 Predicted Genres:\n"
            for i in range(min(3, len(prob_df))):
                genre = prob_df.iloc[i]['Genre']
                prob = prob_df.iloc[i]['Probability']
                top_3_text += f"{i+1}. {genre}: {prob:.1%}\n"
            
            self.top_genres_text.insert("1.0", top_3_text)
            
            # Find similar movies
            similar_movies = self.classifier.find_similar_movies(text)
            if similar_movies:
                similar_text = "Most Similar Movies:\n"
                for title, genre, similarity in similar_movies:
                    similar_text += f"• {title} ({genre}) - Similarity: {similarity:.2f}\n"
                self.similar_movies_text.insert("1.0", similar_text)
            
            # Update visualizations
            self.update_visualizations(prob_df)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def update_visualizations(self, prob_df):
        # Clear previous visualizations
        if self.prob_canvas:
            self.prob_canvas.get_tk_widget().destroy()
        if self.feature_canvas:
            self.feature_canvas.get_tk_widget().destroy()
        
        # Genre probability visualization
        fig_prob, ax_prob = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=prob_df,
            y='Genre',
            x='Probability',
            ax=ax_prob,
            palette='viridis'
        )
        ax_prob.set_title('Genre Probabilities', pad=20)
        ax_prob.set_xlabel('Probability')
        plt.tight_layout()
        
        self.prob_canvas = FigureCanvasTkAgg(fig_prob, self.prob_frame)
        self.prob_canvas.draw()
        self.prob_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Feature importance visualization
        features, importances = self.classifier.get_feature_importance(top_n=10)
        if len(features) > 0 and len(importances) > 0:
            fig_feat, ax_feat = plt.subplots(figsize=(8, 6))
            feat_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            sns.barplot(
                data=feat_df,
                y='Feature',
                x='Importance',
                ax=ax_feat,
                palette='viridis'
            )
            ax_feat.set_title('Top Important Features', pad=20)
            ax_feat.set_xlabel('Importance Score')
            plt.tight_layout()
            
            self.feature_canvas = FigureCanvasTkAgg(fig_feat, self.feature_frame)
            self.feature_canvas.draw()
            self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def clear_results(self):
        """Clear all result fields"""
        self.prediction_label.config(text="Predicted Genre: ")
        self.top_genres_text.delete("1.0", tk.END)
        self.similar_movies_text.delete("1.0", tk.END)
        
        if self.prob_canvas:
            self.prob_canvas.get_tk_widget().destroy()
            self.prob_canvas = None
        if self.feature_canvas:
            self.feature_canvas.get_tk_widget().destroy()
            self.feature_canvas = None
    
    def run(self):
        self.root.mainloop()

def main():
    # Initialize classifier
    classifier = MovieGenreClassifier()
    
    try:
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(current_dir, "Genre Classification Dataset")
        
        # Define file paths
        train_path = os.path.join(dataset_dir, "train_data.txt")
        test_path = os.path.join(dataset_dir, "test_data.txt")
        solution_path = os.path.join(dataset_dir, "test_data_solution.txt")
        
        print(f"Current directory: {current_dir}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Train path: {train_path}")
        print(f"Test path: {test_path}")
        print(f"Solution path: {solution_path}")
        
        # Check if model exists
        if os.path.exists('best_model.pkl'):
            print("Loading existing model...")
            try:
                with open('best_model.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                    classifier.model = saved_data['model']
                    classifier.vectorizer = saved_data['vectorizer']
                    classifier.feature_names = saved_data['feature_names']
                    classifier.genres = saved_data['genres']
                    classifier.train_data = saved_data['train_data']
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Training new model...")
                train_data, test_data, solution_data = classifier.load_data(train_path, test_path, solution_path)
                best_model, best_score = classifier.train_model(train_data, test_data, solution_data)
                
                # Save the model and its components
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump({
                        'model': classifier.model,
                        'vectorizer': classifier.vectorizer,
                        'feature_names': classifier.feature_names,
                        'genres': classifier.genres,
                        'train_data': classifier.train_data
                    }, f)
        else:
            print("Training new model...")
            train_data, test_data, solution_data = classifier.load_data(train_path, test_path, solution_path)
            best_model, best_score = classifier.train_model(train_data, test_data, solution_data)
            
            # Save the model and its components
            with open('best_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': classifier.model,
                    'vectorizer': classifier.vectorizer,
                    'feature_names': classifier.feature_names,
                    'genres': classifier.genres,
                    'train_data': classifier.train_data
                }, f)
        
        # Create and run GUI
        gui = MovieGenreGUI(classifier)
        gui.run()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Current working directory:", os.getcwd())
        raise

if __name__ == "__main__":
    main()
