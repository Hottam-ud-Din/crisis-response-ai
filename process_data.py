import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories datasets and merges them.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on the 'id' column
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Cleans the combined dataframe: splits categories, converts to binary,
    and removes duplicates.
    """
    # 1. Split 'categories' into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # 2. Rename columns using the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]) # Remove the last 2 chars ('-0' or '-1')
    categories.columns = category_colnames
    
    # 3. Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # Binary validation: Ensure values are only 0 or 1 (some datasets have '2')
        categories[column] = categories[column].apply(lambda x: 1 if x >= 1 else 0)
    
    # 4. Replace the original 'categories' column in df with the new columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # 5. Remove duplicates
    df = df.drop_duplicates()
    
    return df

def add_custom_geotags(df):
    """
    BONUS FOR I-APS: Simple rule-based extraction to find Pakistan-specific locations.
    This creates a 'province_mentioned' column.
    """
    def find_location(text):
        text = str(text).lower()
        if 'peshawar' in text or 'kpk' in text or 'khyber' in text:
            return 'KPK'
        elif 'sindh' in text or 'karachi' in text:
            return 'Sindh'
        elif 'punjab' in text or 'lahore' in text:
            return 'Punjab'
        elif 'balochistan' in text or 'quetta' in text:
            return 'Balochistan'
        else:
            return 'Unknown'

    df['province_mentioned'] = df['message'].apply(find_location)
    return df

def save_data(df, database_filename):
    """
    Saves the clean dataset into an SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    # 'replace' ensures we don't duplicate tables if we run this twice
    df.to_sql('disaster_response_table', engine, index=False, if_exists='replace') 

def main():
    print('Loading data...')
    # Assuming your files are named 'messages.csv' and 'categories.csv'
    df = load_data('disaster_messages.csv', 'disaster_categories.csv')

    print('Cleaning data...')
    df = clean_data(df)
    
    print('Adding custom geotags (i-APS Bonus)...')
    df = add_custom_geotags(df)

    print(f'Saving data to DisasterResponse.db...')
    save_data(df, 'DisasterResponse.db')
    
    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()