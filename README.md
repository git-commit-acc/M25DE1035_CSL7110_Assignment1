# Project Gutenberg Text Analysis with PySpark

A comprehensive Big Data project utilizing **PySpark** to perform metadata extraction, NLP-based document similarity (TF-IDF), and temporal graph analysis on the Project Gutenberg corpus.

## 🚀 Project Overview
This project processes raw text data from Project Gutenberg to extract structured insights through three core analytical modules:
1.  **Metadata Profiling:** Statistical analysis of book origins, languages, and titles.
2.  **Semantic Similarity:** Content-based recommendation using manual TF-IDF and Cosine Similarity.
3.  **Influence Networking:** Temporal graph construction to identify "influential" authors within a specific time window.

## 📂 Project Structure
* `q10.py`: Script for metadata extraction and corpus statistics.
* `q11_1.py`: NLP pipeline for text cleaning, TF-IDF vectorization, and pairwise similarity.
* `q12.py`: Script for author influence graph construction and degree centrality analysis.
* `data/`: Local directory containing Project Gutenberg `.txt` files.

---

## 🛠️ Solutions Implemented

### Solution 10: Metadata Extraction & Analysis
Extracts critical book attributes directly from raw text headers using optimized Regular Expressions.
* **Key Metrics:**
    * Distribution of books released per year.
    * Identification of the most common language in the sample.
    * Calculation of average character length for book titles.
* **Implementation:** Utilizes `regexp_extract` to parse structured headers for Title, Release Date, Language, and Encoding.

### Solution 11: Document Similarity (TF-IDF)
Finds semantically similar books by converting text into high-dimensional vectors.
* **NLP Pipeline:**
    1.  **Cleaning:** Strips Gutenberg-specific headers/footers using multi-line regex and removes non-alphabetic characters.
    2.  **Tokenization:** Uses `RegexTokenizer` and `StopWordsRemover` to isolate meaningful keywords.
    3.  **Vectorization:** Manually calculates **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.
    4.  **Similarity:** Implements **Cosine Similarity** math (Dot Product / Magnitudes) to compare all book pairs.
* **Goal:** Identifies the Top 5 most similar books to a target file (e.g., `10.txt`).

### Solution 12: Author Influence Graph
Models a simplified "Influence Network" based on publication timelines.
* **Edge Logic:** A directed edge is created from **Author A → Author B** if Author B published a book within a **5-year window (X=5)** after Author A.
* **Graph Metrics:**
    * **In-Degree:** Identifies "Most Influenced" authors (those whose work followed many contemporaries).
    * **Out-Degree:** Identifies "Most Influential" authors (those whose work preceded many others).

---

## 🔧 Setup & Configuration

### Requirements
* Python 3.x
* PySpark 3.x
* Java 8/11

### Data Pathing
The scripts are configured for a **WSL/Linux environment** accessing a Windows D: drive:
`file:///mnt/d/Installations/coding/HadoopEnv/hadoop/etc/hadoop/guntenberg_data/*.txt`.

### Execution
Run each module individually in your Spark-enabled terminal:
```bash
python q10.py
python q11_1.py
python q12.py
