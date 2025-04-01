import pandas as pd
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

# download stopwords
nltk.data.path.append("../data/input/nltk_data")
from nltk.corpus import stopwords

df = pd.read_csv("../data/input/TMDB_movie_dataset_v11.csv")
non_null_counts = df.count()  # Counts non-null values
null_counts = df.isnull().sum()  # Counts null values

result = pd.DataFrame({"Non-null Count": non_null_counts, "Null Count": null_counts})
print(result)

# # for col in df.columns:
# #     print(f"Column: {col}")
# #     print(df[col].value_counts(), "\n")

### Columns required - title, vote_average, vote_count, revenue, runtime, poster_path, budget, original_language, overview, popularity, release_date
# Drop rows with the above columns = null
df = df.dropna(subset=[
    "title", "vote_average", "vote_count", "revenue", "runtime", "poster_path", "budget", "original_language", "overview", "popularity",
    "release_date"
])

# Drop rows with duplicated title, release_date
df = df.drop_duplicates(subset=['title', 'release_date'], keep='first')
# print(df.isnull().sum())

### Update poster_path and backdrop_path with url
base_url = "https://image.tmdb.org/t/p/w780" #width 780 standardised
df["poster_path"] = base_url + df["poster_path"]
# df["backdrop_path"] = base_url + df["backdrop_path"]

### Convert release_date to year, month, day columns
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month
df["release_day"] = df["release_date"].dt.day


### Text preprocessing
# Combine overview and tagline columns into a single text column
df["text"] = df["overview"] + " " + df["tagline"].fillna("") + " " + df["keywords"].fillna("")

# Define stopwords (you can extend this list as needed)
stop_words = set(stopwords.words("english"))

# Function for cleaning and removing stop words and non-sensible words
def preprocess_text(text):
    words = re.split(r'[,\s]+', text)
    processed_words = []

    for word in words:
        # Convert to lowercase for consistent matching
        word = word.lower()

        # Remove words that are too short (less than 3 characters) or just digits
        if len(word) <= 2 or word.isdigit():
            continue

        # Remove alphanumeric combinations (e.g., abc123, movie42)
        if re.match(r'^[a-zA-Z]*\d+[a-zA-Z]*$', word):
            continue

        # Remove words with numbers and units (e.g., 100ft, 100lbs)
        if re.match(r'^\d+[a-zA-Z]+$', word):  # e.g., 100ft, 20kg, etc.
            continue

        # Remove time-related words (e.g., 12pm, 1am)
        if re.match(r'^\d{1,2}(am|pm)$', word):  # e.g., 12pm, 6am, etc.
            continue

        # Remove common numeric representations (e.g., 100, 1000, 2000)
        if re.match(r'^\d+$', word):  # e.g., 100, 1000, 2000
            continue

        # Remove any word that has no meaningful context (like _ozified_)
        if re.match(r'^[a-zA-Z0-9]*[^\w\s][a-zA-Z0-9]*$', word):  # for special characters or jumbled words
            continue

        # Remove stop words
        if word not in stop_words:
            processed_words.append(word)

    return " ".join(processed_words)

# Apply the text preprocessing
df["processed_text"] = df["text"].apply(preprocess_text)


### Export to csv
# Function to split and save CSV
def split_and_save_csv(df, filename, chunk_size=10000):
    """Splits large DataFrame into smaller chunks and writes them into respective folders with renamed CSVs."""
    base_dir = os.path.dirname(filename)
    folder_name = os.path.basename(filename).split('.')[0]  # Get the name of the base file without extension
    base_dir = os.path.join(base_dir, folder_name)  # Folder for processed_movies or tfidf_movies
    
    # Create the parent folder for the specific file type
    os.makedirs(base_dir, exist_ok=True)

    # Split and save each chunk into the respective folder
    for i, start in enumerate(tqdm(range(0, len(df), chunk_size), desc="Saving chunks")):
        chunk_df = df.iloc[start : start + chunk_size]
        
        # Define the filename for each chunk with _1, _2, _3, etc.
        chunk_filename = os.path.join(base_dir, f"{folder_name}_{i+1}.csv")

        # Write the chunk to its corresponding file
        chunk_df.to_csv(chunk_filename, index=False)
        
        print(f"Saved chunk {i+1} to {chunk_filename}")
        
split_and_save_csv(df, "../data/output/processed_movies.csv")
print("end")

non_null_counts = df.count()  # Counts non-null values
null_counts = df.isnull().sum()  # Counts null values

result = pd.DataFrame({"Non-null Count": non_null_counts, "Null Count": null_counts})
print(result)



# Vectorize using TF-IDF
# vectorizer = TfidfVectorizer(stop_words="english")
# tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

# # Convert the TF-IDF matrix to DataFrame
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
# split_and_save_csv(tfidf_df, "../data/output/tfidf_movies.csv")
# print('end 2')