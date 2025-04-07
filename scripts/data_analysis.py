import os
import json
import sys
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

try:
    # Read CSV files
    df = spark.read.csv(
        full_file_paths,
        header=True,
        inferSchema=True,
        nullValue="NA",
    )
    # Convert columns to numeric types
    df = df.withColumn("runtime", col("runtime").cast("double"))
    df = df.withColumn("revenue", col("revenue").cast("double"))
    df = df.withColumn("budget", col("budget").cast("double"))
    df = df.withColumn("release_year", col("release_year").cast("int"))
    df = df.withColumn("release_month", col("release_month").cast("int"))

    # --- STEP 1: Voting Distribution ---
    df_clean = df.filter(col("vote_count") >= 10)

    df_clean.select("vote_average").groupBy("vote_average").count().orderBy(
        "vote_average", ascending=False
    ).show()

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


except Exception as e:
    print(f"An error occurred: {e}")
    spark.stop()

finally:
    spark.stop()


# # STEP5

# # 載入 ResNet50 模型 (去掉最後一層 classifier)
# base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# # 圖片預處理 function
# def extract_features(img_url):
#     try:
#         response = requests.get(img_url)
#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         img = img.resize((224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)

#         features = base_model.predict(x)
#         return features.flatten()
#     except:
#         return np.zeros((2048,))  # 若讀取失敗則填 0


# # 載入 CSV 檔案
# df = pd.read_csv("processed_movies_1.csv")

# # 只保留 id 和 poster_path 欄位
# poster_df = df[["id", "poster_path"]].dropna()

# # 提取特徵
# feature_list = []
# ids = []

# for idx, row in tqdm(poster_df.iterrows(), total=poster_df.shape[0]):
#     img_url = row["poster_path"]
#     feat = extract_features(img_url)
#     feature_list.append(feat)
#     ids.append(row["id"])

# # 將特徵組成 DataFrame（2048維）
# features_df = pd.DataFrame(feature_list)
# features_df["id"] = ids

# # 儲存為 CSV 或與 Spark 合併
# features_df.to_csv("poster_features_resnet50.csv", index=False)

# # 讀入電影主表
# movies_df = spark.read.csv("processed_movies_1.csv", header=True, inferSchema=True)
# # 讀入 ResNet 特徵表
# resnet_df = spark.read.csv(
#     "poster_features_resnet50.csv", header=True, inferSchema=True
# )

# # 過濾高評分電影（如 vote_average > 8.0）
# high_rating_df = movies_df.filter(col("vote_average") >= 8.0).select("id", "title")

# # 與圖像特徵 join
# high_rating_feat_df = high_rating_df.join(resnet_df, on="id")

# # 將 2048 維特徵轉為 DenseVector
# vector_cols = [c for c in high_rating_feat_df.columns if c not in ["id", "title"]]


# def to_densevec(*vals):
#     return Vectors.dense([float(v) for v in vals])


# to_vec_udf = udf(to_densevec, returnType=DenseVector())
# high_rating_feat_df = high_rating_feat_df.withColumn(
#     "features", to_vec_udf(*[col(c) for c in vector_cols])
# )

# # 將 Spark DataFrame 轉 Pandas
# high_rating_feat_pd = high_rating_feat_df.select("features").toPandas()
# vec_array = np.array([v.toArray() for v in high_rating_feat_pd["features"]])

# # 計算平均向量
# mean_vector = vec_array.mean(axis=0)

# # 找出平均值最高的維度（例如前 10 個）
# top_indices = mean_vector.argsort()[::-1][:10]
# print("Top 10 visual feature dimensions common in high-rated movies (ResNet50 index):")
# for i in top_indices:
#     print(f"Feature[{i}] = {mean_vector[i]:.4f}")


# kmeans = KMeans(k=5, seed=42, featuresCol="features", predictionCol="cluster")
# model = kmeans.fit(high_rating_feat_df)
# clustered = model.transform(high_rating_feat_df)

# # 查看每個 cluster 有哪些電影
# clustered.select("title", "cluster").orderBy("cluster").show(30, truncate=False)

# # STEP6
# mean_vector = np.mean(vec_array, axis=0)
# top_dims = mean_vector.argsort()[::-1][:10]


# kmeans = KMeans(k=5, featuresCol="features", predictionCol="cluster")
# model = kmeans.fit(high_rating_feat_df)
# clustered = model.transform(high_rating_feat_df)
