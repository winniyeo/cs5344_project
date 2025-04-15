import os
import json
import requests
from PIL import Image
from io import BytesIO
import sys
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    floor,
    split,
    explode,
    trim,
    when,
    regexp_replace,
    stddev,
    lower,
    concat_ws,
    desc,
    avg,
    count,
)
from pyspark.ml.feature import (
    VectorAssembler,
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
)
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.ml.clustering import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_project_root():
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.path.dirname(os.getcwd())


# Get project root dynamically
project_root = get_project_root()

# Correct base path
base_path = os.path.join(project_root, "data", "output", "processed_movies")

# List of files
file_list = [f"processed_movies_{i}.csv" for i in range(1, 65)]

# Construct full file paths
full_file_paths = [os.path.join(base_path, filename) for filename in file_list]

# Create Spark Session
spark = (
    SparkSession.builder.appName("MovieAnalysis")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.local.dir", os.path.join(os.getcwd(), "temp_spark"))
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.cleaner.referenceTracking", "true")
    .config("spark.cleaner.periodicGC.interval", "30min")
    .getOrCreate()
)
spark.conf.set("spark.sql.debug.maxToStringFields", 2000)
try:
    df = spark.read.csv(
        full_file_paths,
        header=True,
        inferSchema=True,
        nullValue="NA",
        quote='"',  # optional: handle quoted fields
        escape='"',  # optional: handle quotes inside fields
    )
    # Convert columns to numeric types
    df = df.withColumn("runtime", col("runtime").cast("double"))
    df = df.withColumn("revenue", col("revenue").cast("double"))
    df = df.withColumn("budget", col("budget").cast("double"))
    df = df.withColumn("release_year", col("release_year").cast("int"))
    df = df.withColumn("release_month", col("release_month").cast("int"))
    df = df.withColumn("overview", regexp_replace(col("overview"), '"', ""))
    df = df.withColumn("title", regexp_replace(col("title"), r'[\\/*?:"<>|]', ""))

    # --- STEP 1: Voting Distribution ---
    df_clean = df.filter(col("vote_count") >= 10)

    # Group vote_average into ranges from 1 to 10 using the floor(round down)
    df_grouped = df_clean.withColumn("rating_group", floor(col("vote_average")))

    # Count the number of movies in each rating group.
    rating_dist = (
        df_grouped.groupBy("rating_group")
        .count()
        .orderBy("rating_group", ascending=False)
    )

    # Convert Pandas
    rating_pd = rating_dist.toPandas()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        rating_pd["rating_group"], rating_pd["count"], color="orange", edgecolor="black"
    )

    plt.title("Rating Distribution (vote_count ≥ 100)")
    plt.xlabel("Rating Group")
    plt.ylabel("Number of Movies")
    plt.xticks(rating_pd["rating_group"])
    plt.grid(axis="y")
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 50,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Save the plot into png
    plt.savefig(os.path.join(project_root, "data", "output", "rating_distribution.png"))
    plt.close()

    # --- STEP 2: Filter High Rating Movies (rating >= 7.0) ---
    high_rating_df = df_grouped.filter(col("rating_group") >= 7.0)
    # high_rating_df.filter(col("id") == "754").show()  # test
    # high_rating_df.filter(col("id") == "550").show()  # test
    # --- STEP 3: Correlation Analysis ---
    # Numerical columns
    numerical_cols = [
        "rating_group",
        "runtime",
        "revenue",
        "budget",
        "release_year",
        "release_month",
    ]

    df_merged = high_rating_df.select(["id"] + numerical_cols)

    # Identify valid numerical columns (with non-zero standard deviation)
    other_cols = [c for c in df_merged.columns if c not in ["id"]]
    stddevs_num = (
        df_merged.select([stddev(col(c)).alias(c) for c in other_cols])
        .collect()[0]
        .asDict()
    )
    valid_numerical_cols = [
        k for k, v in stddevs_num.items() if v is not None and v > 0
    ]

    # Assemble features
    assembler = VectorAssembler(inputCols=valid_numerical_cols, outputCol="features")
    df_vector = assembler.transform(df_merged)

    # Calculate Pearson Correlation
    corr_matrix = Correlation.corr(df_vector, "features", "pearson").head()[0]
    corr_array = corr_matrix.toArray()

    def visualize_correlation_matrix(corr_array, feature_names):
        """
        Create an improved heatmap visualization of the correlation matrix
        """
        # Convert to pandas DataFrame
        corr_df = pd.DataFrame(corr_array, index=feature_names, columns=feature_names)

        # Create heatmap with improved readability
        plt.figure(figsize=(16, 14))

        # Use a more perceptually uniform colormap
        sns.heatmap(
            corr_df,
            annot=True,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
            fmt=".2f",
            annot_kws={
                "size": 8,
                "fontweight": "light",
            },
        )

        plt.title(
            "Feature Correlation Heatmap for High-Rated Movies", fontsize=16, pad=20
        )
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        # Save the plot into png
        plt.savefig(
            os.path.join(project_root, "data", "output", "correlation_heatmap.png")
        )
        plt.close()

        # Enhanced correlation reporting
        print("\nSignificant Correlations:")

        correlation_thresholds = {
            "Strong Positive": (0.7, 1.0),
            "Moderate Positive": (0.5, 0.7),
            "Weak Positive": (0.3, 0.5),
            "Strong Negative": (-1.0, -0.7),
            "Moderate Negative": (-0.7, -0.5),
            "Weak Negative": (-0.5, -0.3),
        }

        for correlation_type, (lower, upper) in correlation_thresholds.items():
            print(f"\n{correlation_type} Correlations:")
            found_correlations = False
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    corr_value = corr_df.iloc[i, j]
                    if lower <= corr_value < upper or lower < -corr_value <= upper:
                        print(
                            f"{feature_names[i]} - {feature_names[j]}: {corr_value:.3f}"
                        )
                        found_correlations = True
            if not found_correlations:
                print("No correlations found in this range.")

    # Visualize correlation matrix
    visualize_correlation_matrix(corr_array, valid_numerical_cols)

    spark.catalog.clearCache()

    # --- STEP 4: Analyzing Frequent Words ---
    # Merge keywords and processed_text into one column
    merged_text_df = high_rating_df.withColumn(
        "merged_text", concat_ws(" ", col("keywords"), col("processed_text"))
    )

    # Clean text: Remove non-alphabetic characters and trim spaces
    merged_text_df = (
        merged_text_df.withColumn(
            "cleaned_text",
            regexp_replace(lower(col("merged_text")), "[^a-zA-Z\\s]", ""),
        )
        .withColumn(
            "cleaned_text",
            regexp_replace(col("cleaned_text"), "\\s+", " "),  # Remove extra spaces
        )
        .filter(col("cleaned_text") != "")
    )  # Remove empty rows

    # Tokenize text into words
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    words_data = tokenizer.transform(merged_text_df)

    # Remove stop words (common words like "the", "a", "on")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    filtered_data = stopwords_remover.transform(words_data)

    # Convert words into feature vectors using CountVectorizer
    cv = CountVectorizer(
        inputCol="filtered_words", outputCol="raw_features", vocabSize=500
    )
    cv_model = cv.fit(filtered_data)
    featurized_data = cv_model.transform(filtered_data)

    # Apply TF-IDF transformation
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(featurized_data)
    tfidf_data = idf_model.transform(featurized_data)

    # Extract words with highest TF-IDF scores
    vocab = cv_model.vocabulary
    tfidf_scores = tfidf_data.select(explode(col("filtered_words")).alias("word"))

    # Count word frequencies
    word_counts = tfidf_scores.groupBy("word").count().orderBy(desc("count"))

    # Filter only words in the vocabulary
    word_counts_filtered = word_counts.filter(col("word").isin(vocab))

    # Convert to Pandas and save as CSV
    word_counts_pd = word_counts_filtered.toPandas()
    output_path = os.path.join(
        project_root, "data", "output", "tfidf_frequent_words.csv"
    )
    word_counts_pd.to_csv(output_path, index=False)

    # --- STEP 5: Analyzing Poster Similarity ---
    high_rating_pdf = high_rating_df.toPandas()
    POSTER_FOLDER = "downloaded_posters"
    os.makedirs(POSTER_FOLDER, exist_ok=True)

    # --- STEP 5.1: CNN ---
    # Load ResNet50 model
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Extract feature from image
    def extract_features(img_path):
        try:
            img = image.load_img(img_path, target_size=(780, 780))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            return features.flatten()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return np.zeros(2048)  # fallback in case of error

    # Define download posters function
    def download_image(url, filename):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img.save(filename)
                return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
        return False

    # Download all posters
    for i, row in tqdm(high_rating_pdf.iterrows(), total=len(high_rating_pdf)):
        title = row["title"]
        url = row["poster_path"]
        save_path = os.path.join(POSTER_FOLDER, f"{title}.jpg")
        if not os.path.exists(save_path):
            download_image(url, save_path)

    # Extract features from all images
    data = []
    print("Extracting features...")
    for i, row in tqdm(high_rating_pdf.iterrows(), total=len(high_rating_pdf)):
        title = row["title"]
        img_path = os.path.join(POSTER_FOLDER, f"{title}.jpg")
        if os.path.exists(img_path):
            features = extract_features(img_path)
            data.append([title] + features.tolist())

    # Create DataFrame
    columns = ["title"] + [f"feat_{i}" for i in range(2048)]
    df_features = pd.DataFrame(data, columns=columns)

    # Save as Parquet (for PySpark use)
    df_features.to_parquet("data/output/movie_image_features.parquet", index=False)

    # (Optional) Also save as CSV
    # df_features.to_csv("movie_image_features.csv", index=False)

    # --- STEP 5.2: Image Analysis(PySpark) ---
    # Load saved features back into PySpark
    df_features_spark = spark.read.parquet("data/output/movie_image_features.parquet")

    # Assemble features into a vector
    feature_cols = [f"feat_{i}" for i in range(2048)]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vector = assembler.transform(df_features_spark)

    # STEP 5.2.1: SimilarItems Analysis(KMeans clustering)
    k = 5  # can change k for more/fewer clusters
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
    model = kmeans.fit(df_vector)

    # Assign clusters
    clustered_df = model.transform(df_vector)

    # Show cluster assignments
    clustered_df.select("title", "cluster").show(20, truncate=False)

    # Count movies in each cluster
    clustered_df.groupBy("cluster").count().orderBy("count", ascending=False).show()

    # Visualize KMeans clustering in 2D using PCA
    print("\n=== Visualizing Clusters with PCA ===")
    sample_df = df_features_spark.select([f"feat_{i}" for i in range(2048)]).limit(1000)
    # collect to Pandas
    feature_array = sample_df.toPandas().values
    cluster_labels = (
        clustered_df.select("cluster").limit(1000).toPandas()["cluster"].values
    )

    # Downsizing to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(feature_array)

    # PCA plotting
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.title("PCA Projection of Poster Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(project_root, "data", "output", "cluster_plot.png"))

    # STEP 5.2.2: FrequentItems Analysis
    print("\n=== FrequentItems Analysis ===")

    # Calculate mean for every feature
    mean_vector = np.mean(feature_array, axis=0)
    top_k = 10
    top_indices = np.argsort(mean_vector)[-top_k:][::-1]

    # STEP 5.2.2.1: Top Activated Features in general
    print(f"Top {top_k} Most Frequently Activated Features in High-Rated Posters:")
    for idx in top_indices:
        print(f"Feature {idx} → Mean Activation: {mean_vector[idx]:.4f}")

    # Choose top feature(frequently appearing)and corresponding Top10 movies with this feature
    top_feature_idx = top_indices[0]
    df_features = df_features_spark.toPandas()
    top_posters = (
        df_features[["title", f"feat_{top_feature_idx}"]]
        .sort_values(by=f"feat_{top_feature_idx}", ascending=False)
        .head(10)
    )

    print(f"\nTop 10 Posters with Highest Activation on Feature {top_feature_idx}:")
    print(top_posters)
    # STEP 5.2.2.2: Top5 Activated Features per Cluster
    print("\n=== Top Activated Features per Cluster ===")

    # Merge cluster info
    cluster_labels = clustered_df.select("title", "cluster").toPandas()
    df_features_cluster = df_features.merge(cluster_labels, on="title", how="left")

    feature_cols = [f"feat_{i}" for i in range(2048)]
    top_k = 5  # Top5 Activated Features per Cluster, can change this number

    for cluster_id in sorted(df_features_cluster["cluster"].unique()):
        cluster_df = df_features_cluster[df_features_cluster["cluster"] == cluster_id]
        mean_vec = np.mean(cluster_df[feature_cols].values, axis=0)
        top_idx = np.argsort(mean_vec)[-top_k:][::-1]

        print(f"\n=== Cluster {cluster_id} - Top {top_k} Activated Features ===")
        for idx in top_idx:
            print(f"Feature {idx} → Mean Activation: {mean_vec[idx]:.4f}")

    print("\n=== Top 3 Movies for Each Top Feature per Cluster ===")

    top_k_features = 3  # Top3 Activated Features per Cluster, can change this number
    top_k_movies = (
        3  # Top3 movies for each features in Top3 per Cluster, can change this number
    )

    for cluster_id in sorted(df_features_cluster["cluster"].unique()):
        print(f"\n=== Cluster {cluster_id} ===")

        cluster_df = df_features_cluster[df_features_cluster["cluster"] == cluster_id]
        mean_vec = np.mean(cluster_df[feature_cols].values, axis=0)

        # Find top features that have top activation value per cluster
        top_feat_idx = np.argsort(mean_vec)[-top_k_features:][::-1]

        for feat_idx in top_feat_idx:
            feat_col = f"feat_{feat_idx}"
            top_movies = (
                cluster_df[["title", feat_col]]
                .sort_values(by=feat_col, ascending=False)
                .head(top_k_movies)
            )

            print(
                f"\nTop {top_k_movies} movies for Feature {feat_idx} (avg activation: {mean_vec[feat_idx]:.4f}):"
            )
            for _, row in top_movies.iterrows():
                print(f"- {row['title']} → {feat_col}: {row[feat_col]:.4f}")

    # Save result to CSV (optional)
    clustered_df.select("title", "cluster").toPandas().to_csv(
        os.path.join(
            project_root, "data", "output", "clustered_high_rating_movies.csv"
        ),
        index=False,
    )

    print("KMeans clustering complete and results saved!")
    # Join cluster info back to original high_rating_df metadata
    # Ensure 'title' is available in both DataFrames and properly cleaned/trimmed
    clustered_with_meta = clustered_df.join(
        high_rating_df.select("title", "rating_group", "genres"), on="title", how="left"
    )

    # Analyze average rating per cluster
    print("\n=== Average Rating Group per Cluster ===")
    clustered_with_meta.groupBy("cluster").agg(
        avg("rating_group").alias("avg_rating_group")
    ).orderBy("avg_rating_group", ascending=False).show()

    # Analyze genre distribution per cluster
    print("\n=== Top Genres per Cluster ===")
    # Explode comma-separated genres into rows
    genre_exploded = (
        clustered_with_meta.withColumn("genre", explode(split(col("genres"), ",")))
        .withColumn("genre", trim(col("genre")))
        .filter(col("genre").isNotNull() & (col("genre") != ""))
    )

    # Count number of times each genre appears in each cluster
    genre_counts = genre_exploded.groupBy("cluster", "genre").agg(
        count("*").alias("count")
    )

    # Show top 5 genres per cluster
    genre_window = Window.partitionBy("cluster").orderBy(col("count").desc())
    top_genres_per_cluster = genre_counts.withColumn(
        "row_num", row_number().over(genre_window)
    ).filter(col("row_num") <= 5)
    top_genres_per_cluster.select("cluster", "genre", "count").orderBy(
        "cluster", "count", ascending=False
    ).show(50, truncate=False)

except Exception as e:
    print(f"An error occurred: {e}")
    spark.stop()

finally:
    spark.stop()
