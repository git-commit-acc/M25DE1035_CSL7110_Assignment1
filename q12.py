import time
from pyspark.sql import functions as f

# ---------------------------------------------------------
# 1. PREPROCESSING: Metadata Extraction
# ---------------------------------------------------------
input_path = "file:///mnt/d/Installations/coding/HadoopEnv/hadoop/etc/hadoop/guntenberg_data/*.txt"
raw = spark.sparkContext.wholeTextFiles(input_path)
books_df = raw.toDF(["path", "text"]) \
              .withColumn("file_name", f.regexp_extract("path", r"([^/\\\\]+)$", 1))

# Refined Regex for Author and Release Year
author_re = r"(?mi)^Author:\s*(.+?)\s*$"
year_re = r"(?mi)^Release Date:\s*.*?(\b(18|19|20)\d{2}\b)"

# Create a clean DataFrame of Authors and Years
authors_meta = books_df.withColumn("author", f.nullif(f.trim(f.regexp_extract("text", author_re, 1)), f.lit(""))) \
                       .withColumn("year", f.regexp_extract("text", year_re, 1).cast("int")) \
                       .filter((f.col("author").isNotNull()) & (f.col("year").isNotNull())) \
                       .select("author", "year") \
                       .distinct() # One entry per author per year

print("\n[Extracted Author and Release Year Metadata]")
authors_meta.show(15, truncate=False)
time.sleep(5)

# ---------------------------------------------------------
# 2. INFLUENCE NETWORK CONSTRUCTION
# ---------------------------------------------------------
# Define time window X
X = 5 

# Self-join to find pairs where Author B published within X years AFTER Author A
# This creates a directed edge: Author A -> Author B (A influenced B)
a = authors_meta.alias("a")
b = authors_meta.alias("b")

influence_edges = a.join(b, 
    on=((f.col("a.author") != f.col("b.author")) & 
        (f.col("b.year") > f.col("a.year")) & 
        (f.col("b.year") <= f.col("a.year") + f.lit(X))),
    how="inner"
).select(f.col("a.author").alias("influencer"), f.col("b.author").alias("influenced")).distinct()

print(f"\n[Influence Network Edges (A potentially influenced B, X={X} years)]")
influence_edges.show(15, truncate=False)
time.sleep(5)

# ---------------------------------------------------------
# 3. ANALYSIS: In-Degree and Out-Degree
# ---------------------------------------------------------
# Out-degree: Number of authors they potentially influenced
out_degree = influence_edges.groupBy("influencer") \
                            .agg(f.count("influenced").alias("out_degree")) \
                            .withColumnRenamed("influencer", "author")

# In-degree: Number of authors who potentially influenced them
in_degree = influence_edges.groupBy("influenced") \
                           .agg(f.count("influencer").alias("in_degree")) \
                           .withColumnRenamed("influenced", "author")

# Combine results
degree_df = out_degree.join(in_degree, on="author", how="full_outer").na.fill(0)

print("\n[Top 5 Authors with Highest IN-Degree (Most Influenced)]")
degree_df.orderBy(f.desc("in_degree")).show(5, truncate=False)
time.sleep(5)

print("\n[Top 5 Authors with Highest OUT-Degree (Most Influential)]")
degree_df.orderBy(f.desc("out_degree")).show(5, truncate=False)
time.sleep(5)

# Final Count check
print(f"Total nodes (authors): {authors_meta.select('author').distinct().count()}")
print(f"Total directed edges in network: {influence_edges.count()}")

