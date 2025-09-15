# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load datasets
books = pd.read_csv('Books.csv', encoding='latin-1')
users = pd.read_csv('Users.csv', encoding='latin-1')
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')

# Display dataset shapes
print("Books:", books.shape)
print("Users:", users.shape)
print("Ratings:", ratings.shape)

# Display first few rows of each dataset
print("\nBooks:\n", books.head())
print("\nUsers:\n", users.head())
print("\nRatings:\n", ratings.head())

# Check for missing values
print("\nMissing Values in Books:\n", books.isnull().sum())
print("\nMissing Values in Users:\n", users.isnull().sum())
print("\nMissing Values in Ratings:\n", ratings.isnull().sum())

# Drop books with missing titles or authors
books.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
users.dropna(subset=['Location'], inplace=True)
ratings.dropna(inplace=True)

# Merge datasets
merged = ratings.merge(books, on='ISBN')
print("\nMerged Dataset:\n", merged.head())

# ------------------------------
# Top-Rated Books (with >= 50 ratings)
# ------------------------------
book_rating_count = merged.groupby('Book-Title').count()['Book-Rating']
popular_books = merged.groupby('Book-Title')['Book-Rating'].mean()
ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')


popular_books_df = pd.DataFrame({
    'AverageRating': popular_books,
    'RatingCount': book_rating_count
})

popular_books_df = popular_books_df[popular_books_df['RatingCount'] >= 50]
top_books = popular_books_df.sort_values('AverageRating', ascending=False).head(10)

print("\nTop-Rated Books:\n", top_books)

# Plot top-rated books
sns.barplot(x='AverageRating', y=top_books.index, data=top_books)
plt.title('Top 10 Highest Rated Books (with >=50 ratings)')
plt.xlabel('Average Rating')
plt.ylabel('Book Title')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/plots/top_rated_books.png')
plt.close()

#box plot
top_titles = top_books.index.tolist()
subset = merged[merged['Book-Title'].isin(top_titles)]
# Assuming 'subset' is already defined with top books
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='Book-Title',
    y='Book-Rating',
    data=subset,
    palette='Paired',
    linewidth=2.5
)
plt.title('Rating Spread of Top 10 Books', fontsize=14)
plt.xlabel('Book Title', fontsize=12)
plt.ylabel('Book Rating', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/plots/top_10_boxplot.png')
plt.close()

# ------------------------------
# Rating Distribution
# ------------------------------
sns.countplot(x='Book-Rating', data=ratings)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('static/plots/rating_distribution.png')
plt.close()


# Calculate rating count and average rating per book
book_stats = merged.groupby('Book-Title')['Book-Rating'].agg(['mean', 'count']).reset_index()
book_stats.columns = ['Book-Title', 'AverageRating', 'RatingCount']

# Scatter plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=book_stats, x='RatingCount', y='AverageRating', alpha=0.6)
plt.title('Average Rating vs Number of Ratings per Book')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.xscale('log')
plt.tight_layout()
plt.savefig('static/plots/rating_vs_count.png')
plt.close()


# ------------------------------
# Age Distribution of Users
# ------------------------------
users = users[(users['Age'] > 5) & (users['Age'] < 100)]  # filter out unrealistic ages
sns.histplot(users['Age'], bins=30, kde=True)
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('static/plots/age_distribution.png')
plt.close()


#pie chart
# Load the dataset
users = pd.read_csv('Users.csv', encoding='latin-1')

# Count occurrences of each location
location_counts = users['Location'].value_counts()

# Select the top 10 locations
top_locations = location_counts.head(10)

# Now plot the pie chart
top_locations.plot.pie(autopct='%1.1f%%', figsize=(8, 8), title='Top 10 User Locations')
plt.ylabel('')
plt.tight_layout()
plt.savefig('static/plots/top_locations.png')
plt.close()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset with memory-safe warning handled
books = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')
users = pd.read_csv('Users.csv', encoding='latin-1')

# Merge books with ratings
merged = ratings.merge(books, on='ISBN')

# Keep only relevant columns
merged = merged[['Book-Title', 'Book-Author', 'Book-Rating']]

# Optional: drop nulls
merged.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)

# Limit to most rated books
popular_books = merged.groupby('Book-Title').count().sort_values('Book-Rating', ascending=False).head(5000).reset_index()
filtered = merged[merged['Book-Title'].isin(popular_books['Book-Title'])]

# Drop duplicates
filtered = filtered.drop_duplicates(subset='Book-Title')

# Combine title and author
filtered['content'] = filtered['Book-Title'] + ' ' + filtered['Book-Author']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered['content'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Reset index for lookup
filtered = filtered.reset_index(drop=True)

# Recommend similar books
def recommend_books(title, cosine_sim=cosine_sim, df=filtered):
    indices = pd.Series(df.index, index=df['Book-Title'].str.lower())
    title = title.lower()

    if title not in indices:
        return f" Book titled '{title}' not found in dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]

    return df[['Book-Title', 'Book-Author']].iloc[book_indices]

#  Example Usage
print(filtered['Book-Title'].sample(10).to_list())
search_book = "The End of the Pier"
print(f"\n Similar books to '{search_book}':\n")
result = recommend_books(search_book)
if isinstance(result, str):
    print(result)
else:
    print(result.to_string(index=False))


