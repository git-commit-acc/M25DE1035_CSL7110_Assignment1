from pyspark.sql.functions import col, regexp_extract, avg, length, desc

# 1. Load rawdata into Spark
path = "file:///mnt/d/Installations/coding/HadoopEnv/hadoop/etc/hadoop/guntenberg_data/*.txt"

raw_rdd = spark.sparkContext.wholeTextFiles(path)

if raw_rdd.isEmpty():
    print("Error: No files found at the specified path. Check your folder path.")
else:
    books_df = raw_rdd.map(lambda x: (x[0].split("/")[-1], x[1])).toDF(["file_name", "text"]).limit(25)
    print("\n[Books DataFrame Preview]")
    books_df.show(5, truncate=80)

    # 2. Metadata Extraction (use RAW STRINGS for regex)
    title_re = r"Title:\s*(.+)"
    date_re = r"Release Date:\s*.*?(\d{4})"
    lang_re = r"Language:\s*(.+)"
    enc_re = r"Character set encoding:\s*(.+)"

    metadata_df = books_df.withColumn("title", regexp_extract("text", title_re, 1)) \
                          .withColumn("release_year", regexp_extract("text", date_re, 1).cast("int")) \
                          .withColumn("language", regexp_extract("text", lang_re, 1)) \
                          .withColumn("encoding", regexp_extract("text", enc_re, 1))

    metadata_df.cache()
    print("\n[Extracted Metadata Preview]")
    metadata_df.select("file_name","title","release_year","language","encoding").show(10, truncate=60)

    # 3. Analysis Results
    print("\n" + "="*40)
    print("ANALYSIS RESULTS (Question 10)")
    print("="*40)

    print("\n[Books Released Each Year]")
    metadata_df.filter(col("release_year").isNotNull()) \
               .groupBy("release_year").count() \
               .orderBy("release_year").show()

    print("\n[Most Common Language]")
    metadata_df.filter(col("language") != "") \
               .groupBy("language").count() \
               .orderBy(desc("count")).limit(1).show()

    print("\n[Average Character Length of Titles]")
    metadata_df.filter(col("title") != "") \
               .select(avg(length(col("title"))).alias("avg_title_chars")).show()
