import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'omw-1.4', 'punkt_tab'])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Loads data from the SQLite database and separates features (X) from targets (Y).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response_table', engine)
    
    # X is the raw text message
    X = df['message']
    
    # Y is all the category columns (we exclude id, message, original, genre, province_mentioned)
    # The province_mentioned is metadata, not a target to predict here.
    Y = df.drop(['id', 'message', 'original', 'genre', 'province_mentioned'], axis=1)
    
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Normalizes, tokenizes, and lemmatizes text string.
    """
    # 1. Tokenize (split into words)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        # 2. Lemmatize (convert 'running' -> 'run', 'tables' -> 'table')
        # 3. Normalize (lowercase, strip whitespace)
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    Builds the Machine Learning Pipeline.
    This is the specific architecture i-APS will care about.
    """
    pipeline = Pipeline([
        # Step 1: Count occurrences of words
        ('vect', CountVectorizer(tokenizer=tokenize)),
        
        # Step 2: Transform counts to TF-IDF (weighs unique words higher)
        ('tfidf', TfidfTransformer()),
        
        # Step 3: MultiOutput Classifier
        # We wrap RandomForest because we need to predict 36 categories at once.
        # n_jobs=-1 uses all your CPU cores to train faster.
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints precision, recall, and f1-score for each category.
    """
    Y_pred = model.predict(X_test)
    
    # Calculate accuracy across all 36 categories
    # converting to DataFrame for easier reporting
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    
    print("\n--- Model Performance Report ---")
    # We print a report for a few key columns to keep the output readable
    for i, col in enumerate(category_names):
        # limiting print to first 5 categories for brevity in the console
        if i < 5: 
            print(f"Category: {col}")
            print(classification_report(Y_test[col], Y_pred_df[col]))
            print("-" * 60)

def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    database_filepath = 'DisasterResponse.db'
    model_filepath = 'classifier.pkl'

    print(f'Loading data from {database_filepath}...')
    X, Y, category_names = load_data(database_filepath)
    
    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print('Building model pipeline...')
    model = build_model()
    
    print('Training model (this may take a few minutes)...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)
    
    print(f'Saving model to {model_filepath}...')
    save_model(model, model_filepath)
    
    print('Trained model saved!')

if __name__ == '__main__':
    main()