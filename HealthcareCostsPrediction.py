import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class BookRecommender:
    def __init__(self, books_file='BX-Books.csv', ratings_file='BX-Book-Ratings.csv'):
        """Initialize the recommender with dataset file paths and load data."""
        self.books_file = books_file
        self.ratings_file = ratings_file
        self.df_books = None
        self.df_ratings = None
        self.pivot_table = None
        self.sparse_matrix = None
        self.model = None
        self.load_data()
        self.preprocess_data()
        self.train_model()

    def load_data(self):
        """Load the Book-Crossings dataset into dataframes."""
        self.df_books = pd.read_csv(
            self.books_file,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=['isbn', 'title', 'author'],
            usecols=['isbn', 'title', 'author'],
            dtype={'isbn': 'str', 'title': 'str', 'author': 'str'}
        )

        self.df_ratings = pd.read_csv(
            self.ratings_file,
            encoding="ISO-8859-1",
            sep=";",
            header=0,
            names=['user', 'isbn', 'rating'],
            usecols=['user', 'isbn', 'rating'],
            dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'}
        )

    def preprocess_data(self):
        """Filter users and books, merge data, and create pivot table."""
        # Filter users with at least 200 ratings
        user_counts = self.df_ratings['user'].value_counts()
        self.df_ratings = self.df_ratings[self.df_ratings['user'].isin(user_counts[user_counts >= 200].index)]

        # Filter books with at least 100 ratings
        book_counts = self.df_ratings['isbn'].value_counts()
        self.df_ratings = self.df_ratings[self.df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

        # Merge ratings with book titles
        df = self.df_ratings.merge(self.df_books[['isbn', 'title']], on='isbn')

        # Create a pivot table of users vs. books with ratings as values
        self.pivot_table = df.pivot_table(index='title', columns='user', values='rating').fillna(0)

        # Convert to sparse matrix for efficiency
        self.sparse_matrix = csr_matrix(self.pivot_table.values)

    def train_model(self):
        """Train the KNN model using cosine distance."""
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
        self.model.fit(self.sparse_matrix)

    def get_recommends(self, book=""):
        """Return 5 recommended books with distances for the given book title."""
        # Check if the book exists in the dataset
        if book not in self.pivot_table.index:
            print(f"Book '{book}' not found in dataset")
            return [book, []]

        # Get the index of the book in the pivot table
        book_idx = self.pivot_table.index.get_loc(book)

        # Find the 6 nearest neighbors (including the book itself)
        distances, indices = self.model.kneighbors(self.sparse_matrix[book_idx], n_neighbors=6)

        # Prepare the list of recommended books with distances
        recommended_books = []
        for i in range(1, 6):  # Get top 5 neighbors (skip the book itself)
            neighbor_idx = indices[0][i]
            neighbor_title = self.pivot_table.index[neighbor_idx]
            neighbor_distance = float(distances[0][i])
            recommended_books.append([neighbor_title, neighbor_distance])

        # Sort by distance (descending) to match test case
        recommended_books = sorted(recommended_books, key=lambda x: x[1], reverse=True)

        # Debug: Print recommendations
        print(f"Recommendations for '{book}': {recommended_books}")

        return [book, recommended_books]


def test_book_recommendation():
    """Test the recommendation function for the challenge."""
    recommender = BookRecommender()
    test_pass = True
    recommends = recommender.get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
    for i in range(2):  # Check only the first two books for exact matches
        if recommends[1][i][0] not in recommended_books:
            print(f"Book {recommends[1][i][0]} not in expected list: {recommended_books}")
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            print(
                f"Distance {recommends[1][i][1]} for {recommends[1][i][0]} not within 0.05 of {recommended_books_dist[i]}")
            test_pass = False
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")


if __name__ == "__main__":
    test_book_recommendation()