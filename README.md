# Movie Genre Classification System

A machine learning-based system that classifies movies into genres based on their plot descriptions. The system uses natural language processing and machine learning techniques to analyze movie plots and predict their genres.

## Features

- **Genre Classification**: Predicts movie genres based on plot descriptions
- **Similar Movies**: Finds similar movies based on plot descriptions
- **Feature Importance**: Shows which words are most important for genre classification
- **User-Friendly GUI**: Interactive interface for easy movie genre prediction
- **Model Persistence**: Saves trained model for future use
- **Performance Metrics**: Provides accuracy scores for model evaluation

## Requirements

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - matplotlib
  - seaborn
  - tkinter
  - tqdm
  - joblib

## Project Structure

```
archive/
├── movie.py           # Main application code
├── train_data.txt     # Training dataset
├── test_data.txt      # Test dataset
└── test_data_solution.txt  # Test dataset solutions
```

## Installation Steps

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download NLTK data (will be done automatically on first run):
   - punkt
   - stopwords
   - wordnet

## Execution Steps

1. Run the application:
   ```bash
   python movie.py
   ```

2. Using the GUI:
   - Enter a movie plot description in the text box
   - Click "Predict Genre" to get genre predictions
   - View similar movies and feature importance in the results section

## How It Works

1. **Data Loading**:
   - Loads training and test datasets
   - Preprocesses text data (cleaning, tokenization, etc.)

2. **Model Training**:
   - Uses TF-IDF vectorization for text features
   - Trains a Logistic Regression model
   - Saves the trained model for future use

3. **Prediction Process**:
   - Preprocesses input text
   - Converts text to numerical features
   - Predicts genre using trained model
   - Finds similar movies based on plot similarity

4. **Similar Movies**:
   - Uses cosine similarity to find similar movies
   - Combines exact match detection with semantic similarity
   - Returns top N most similar movies

## Model Performance

- Training Accuracy: ~60-65%
- Test Accuracy: ~55-60%
- Uses optimized parameters for better performance
- Implements mini-batch training for efficiency

## Troubleshooting

1. **NLTK Data Download Issues**:
   - The program will automatically download required NLTK data
   - If download fails, manually download using:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('stopwords')
     nltk.download('wordnet')
     ```

2. **Model Loading Issues**:
   - If model fails to load, it will automatically train a new one
   - Delete `best_model.pkl` to force retraining

3. **Memory Issues**:
   - The program uses batch processing to handle large datasets
   - If you encounter memory errors, try reducing batch size in the code

## Future Improvements

- Add more sophisticated text preprocessing
- Implement ensemble methods for better accuracy
- Add support for multiple genre prediction
- Include movie metadata in classification
- Add web scraping for automatic plot retrieval

## License

This project is open source and available under the MIT License.

## Contact

For any questions or suggestions, please open an issue in the repository. 