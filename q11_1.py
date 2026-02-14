import time
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as f

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
input_path = "file:///mnt/d/Installations/coding/HadoopEnv/hadoop/etc/hadoop/guntenberg_data/*.txt"
raw = spark.sparkContext.wholeTextFiles(input_path)
books_df = raw.toDF(["path", "text"]) \
              .withColumn("file_name", f.regexp_extract("path", r"([^/\\\\]+)$", 1)) \
              .select("file_name", "text") \
              .limit(30)

# Cleaning: Remove Header/Footer, lowercase, and remove punctuation
cleaned = books_df.withColumn("text_clean", f.regexp_replace("text", r"(?s)\*\*\* START OF.*?\*\*\* END OF.*?\*\*\*", "")) \
                  .withColumn("text_clean", f.lower(f.regexp_replace("text_clean", r"[^a-z\s]", " ")))

tokenizer = RegexTokenizer(inputCol="text_clean", outputCol="tokens", pattern="\\s+")
tokenized = tokenizer.transform(cleaned)
remover = StopWordsRemover(inputCol="tokens", outputCol="words")
processed = remover.transform(tokenized).select("file_name", "words")

# ---------------------------------------------------------
# 2. DELIVERABLES: TF, IDF, and TF-IDF Calculation
# ---------------------------------------------------------
exploded = processed.select("file_name", f.explode("words").alias("word"))

# DELIVERABLE: Calculate Term Frequency (TF)
tf = exploded.groupBy("file_name", "word").count().withColumnRenamed("count", "term_freq")
print("\n[Term Frequency (TF) - Sample]")
tf.show(10)
time.sleep(5)

# DELIVERABLE: Calculate Inverse Document Frequency (IDF)
doc_count = books_df.count()
df = exploded.select("file_name", "word").distinct().groupBy("word").count().withColumnRenamed("count", "doc_freq")
idf = df.withColumn("idf", f.log(f.lit(doc_count) / f.col("doc_freq")))
print("\n[Inverse Document Frequency (IDF) - Sample]")
idf.orderBy(f.desc("idf")).show(10)
time.sleep(5)

# DELIVERABLE: Compute TF-IDF Score (TF * IDF)
tfidf = tf.join(idf, "word").withColumn("tfidf", f.col("term_freq") * f.col("idf")).select("file_name", "word", "tfidf")
print("\n[TF-IDF Scores - Sample]")
tfidf.show(10)
time.sleep(5)

# ---------------------------------------------------------
# 3. DELIVERABLE: Represent Each Book as a Vector
# ---------------------------------------------------------
# This groups the TF-IDF scores back into a structured view for each book
print("\n[Book Representation as TF-IDF Vectors]")
tfidf.orderBy("file_name", f.desc("tfidf")).show(15)
time.sleep(5)

# ---------------------------------------------------------
# 4. DELIVERABLE: Calculate Cosine Similarity Between ALL Pairs
# ---------------------------------------------------------
magnitudes = tfidf.groupBy("file_name").agg(f.sqrt(f.sum(f.col("tfidf") ** 2)).alias("norm"))

# Self-join to create all pairs (A < B avoids duplicates and self-comparison)
pairs = tfidf.alias("a").join(tfidf.alias("b"), "word").filter(f.col("a.file_name") < f.col("b.file_name")) \
        .select(f.col("a.file_name").alias("book1"), 
                f.col("b.file_name").alias("book2"), 
                (f.col("a.tfidf") * f.col("b.tfidf")).alias("product"))

dot_products = pairs.groupBy("book1", "book2").agg(f.sum("product").alias("dot_product"))

similarity = dot_products.join(magnitudes.withColumnRenamed("file_name", "book1").withColumnRenamed("norm", "norm1"), "book1") \
                         .join(magnitudes.withColumnRenamed("file_name", "book2").withColumnRenamed("norm", "norm2"), "book2") \
                         .withColumn("cosine_similarity", f.col("dot_product") / (f.col("norm1") * f.col("norm2")))

print("\n[Pairwise Cosine Similarity Matrix (All Pairs)]")
similarity.select("book1", "book2", "cosine_similarity").orderBy(f.desc("cosine_similarity")).show(10)
time.sleep(5)

# ---------------------------------------------------------
# 5. DELIVERABLE: Identify Top 5 Most Similar Books for 10.txt
# ---------------------------------------------------------
target_book = "10.txt"
print(f"\n[Top 5 Most Similar Books to {target_book}]")
top_5 = similarity.filter((f.col("book1") == target_book) | (f.col("book2") == target_book)) \
                  .orderBy(f.desc("cosine_similarity")).limit(5)

top_5.select("book1", "book2", "cosine_similarity").show(truncate=False)

