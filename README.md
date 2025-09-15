# Book Recommendation System & Exploratory Data Analysis

This project performs an in-depth Exploratory Data Analysis (EDA) on the **Book-Crossing dataset** to uncover insights about user ratings, book popularity, and user demographics. Based on this analysis, it also implements a **content-based book recommendation system** that suggests similar books based on their title and author.

## 📋 Features

  * **Data Cleaning and Preprocessing:** Loads and cleans the `Books`, `Users`, and `Ratings` datasets to prepare them for analysis.
  * **Exploratory Data Analysis (EDA):** Investigates and visualizes key aspects of the data, including:
      * Top 10 highest-rated books.
      * Distribution of all book ratings.
      * Relationship between a book's average rating and its number of ratings.
      * Age and location distribution of users.
  * **Content-Based Recommendation System:**
      * Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert book titles and authors into numerical feature vectors.
      * Computes the **Cosine Similarity** between books to find the most similar ones.
      * Provides a function to recommend the top 10 most similar books for a given book title.
  * **Visualization:** Generates and saves multiple plots to a `static/plots/` directory for easy review.

## 📊 Visualizations

The script generates the following visualizations to summarize the findings of the EDA:

| Top 10 Highest Rated Books                               | Rating Distribution                                |
| -------------------------------------------------------- | -------------------------------------------------- |
|      |  |
| **Top 10 User Locations** | **User Age Distribution** |
|          |  |
| **Rating Spread of Top Books** | **Average Rating vs. Number of Ratings** |
|  |  |

## 🤖 How the Recommendation System Works

The recommendation engine is **content-based**, which means it recommends items based on their properties rather than on user ratings.

1.  **Content Definition:** For each book, a "content" string is created by combining its **title** and **author**.
2.  **Vectorization (TF-IDF):** The `TfidfVectorizer` from Scikit-learn converts this collection of text content into a matrix of TF-IDF features. This process gives more weight to words that are important to a specific book's content but less common across all books.
3.  **Similarity Calculation (Cosine Similarity):** The `cosine_similarity` metric is used to calculate the similarity score between all pairs of books based on their TF-IDF vectors. A higher score means the books are more similar in content.
4.  **Recommendation:** When you input a book title, the system finds its vector and returns the top 10 books with the highest cosine similarity scores.


## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

  * Python 3.7+
  * pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kruthi349/YOUR_REPOSITORY_NAME.git
    cd Book-Recommendation-System-Exploratory-Data-Analysis
    ```

2.  **Download the Dataset:**
    Make sure you have the required CSV files in the project's root directory:
    As the file Books.csv is large, it is not uploaded here.get the similar dataset from kaggle

      * `Books.csv`
      * `Users.csv`
      * `Ratings.csv`


### Execution

Run the script from the command line. It will perform the analysis, save the plots, and print a sample recommendation to the console.

```bash
python recommendation.py
```

-----

## ✨ Example Recommendation

The script includes an example of how to use the recommendation function. For instance, if we search for books similar to **"The End of the Pier"**:

**Output:**

```
 Similar books to 'The End of the Pier':

                 Book-Title       Book-Author
    The Hundred Secret Senses     Amy Tan
                    The Pilot     Robert P. Davis 
                   The Jester     James Patterson
                   The Broker     John Grisham
               The Last Juror     John Grisham
        The Last Don: A Novel     Mario Puzo
                   The Street     Ann Petry
                 The Alienist     Caleb Carr
                The Testament     John Grisham
          The Prince of Tides     Pat Conroy


## 🛠️ Technologies Used

  * **Python**
  * **Pandas:** For data manipulation and analysis.
  * **Matplotlib & Seaborn:** For data visualization.
  * **Scikit-learn:** For implementing the TF-IDF and Cosine Similarity models.
