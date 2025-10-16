# Book Recommendation Engine using KNN

This project implements a book recommendation system using the K-Nearest Neighbors (KNN) algorithm, developed as part of the freeCodeCamp Machine Learning with Python certification. The goal is to create a recommendation engine using scikit-learn’s NearestNeighbors in Google Colab that recommends five similar books based on user ratings for a given book title, using the Book-Crossings dataset with 1.1 million ratings of 270,000 books by 90,000 users.

---

👨‍💻 **Author**: Ali Toori – Full-Stack Python Developer  
📺 **YouTube**: [@AliToori](https://youtube.com/@AliToori)  
💬 **Telegram**: [@AliToori](https://t.me/@AliToori)  
📂 **GitHub**: [Github.com/AliToori](https://github.com/AliToori)

---

### Project Overview
The project involves:
1. Loading and preprocessing the Book-Crossings dataset, filtering out users with fewer than 200 ratings and books with fewer than 100 ratings to ensure statistical significance.
2. Using NearestNeighbors from scikit-learn to build a model that measures the “closeness” of books based on user ratings.
3. Creating a get_recommends function that takes a book title as input and returns a list containing the input title and a nested list of five recommended books with their distances from the input book.
4. Ensuring the model meets the challenge requirements by passing the provided test case, which checks the recommendations for "The Queen of the Damned (Vampire Chronicles (Paperback))".
5. Optionally visualizing the dataset to understand rating distributions (not implemented in the core solution but mentioned as an option).

Example output for `get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")`:

``` python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
````
---

### [Google Colab Project Link](https://colab.research.google.com/drive/1YhqiUuH22rZCzQpfbL8msT8cHZ4J_uGR#scrollTo=Xe7RXH3N3CWU)

---

## 🛠 Tech Stack
* Language: Python 3.8+
* Libraries:
  * scikit-learn (for KNN with NearestNeighbors)
  * Pandas (for data preprocessing and manipulation)
  * NumPy (for numerical operations)
  * Matplotlib/Seaborn (optional, for dataset visualization)
* Tools:
  * Google Colab for development, training, and testing (with GPU support)
  * GitHub for version control (optional, if you export the notebook)

---

## 📂 Project Structure
The project is a single Google Colab notebook (fcc_book_recommendation_knn.ipynb) with cells for:
* Importing libraries (scikit-learn, Pandas, etc.)
* Loading and preprocessing the Book-Crossings dataset
* Filtering users (<200 ratings) and books (<100 ratings)
* Building the KNN model using NearestNeighbors
* Defining the get_recommends function
* Testing the model with the provided test cell

Dataset structure:
```bash
Books.csv: Contains book metadata (ISBN, title, etc.)
Ratings.csv: Contains user ratings (User-ID, ISBN, rating)
Users.csv: Contains user information (User-ID, etc.)
```

---

## Usage
1. Open the provided Colab notebook: https://colab.research.google.com/github/freeCodeCamp/boilerplate-book-recommendation-engine/blob/master/fcc_book_recommendation_knn.ipynb
2. Save a copy to your Google Drive (**File > Save a copy in Drive**).
3. Enable GPU for faster training (**Runtime > Change runtime type > GPU**).
4. Run all cells sequentially:
    - Cells 1-3: Import libraries and load the Book-Crossings dataset.
    - Cell 4: Preprocess data (filter users and books based on rating thresholds).
    - Cell 5: Create a pivot table of user-book ratings and fit the KNN model.
    - Cell 6: Define the get_recommends function to return similar books.
    - Cell 7: Test the function with the provided test case.
5. If the test fails, debug the get_recommends function by checking:
    - Data filtering logic (thresholds for users and books)
    - KNN model parameters (e.g., distance metric, number of neighbors)
    - Handling of edge cases (e.g., book not found in dataset)
---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository (if you export the notebook to GitHub): https://github.com/AliToori/Book-Recommendation-Engine-KNN
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.
Alternatively, share an updated Colab notebook link via GitHub issues or Telegram.

---

## 🙏 Acknowledgments
- Built as part of the [freeCodeCamp Machine Learning with Python](https://www.freecodecamp.org/learn/machine-learning-with-python) certification.
- Uses TensorFlow/Keras for model development and Google Colab for cloud-based execution.
- Special thanks to freeCodeCamp for providing the challenge framework and dataset.

## 🆘 Support
For questions, issues, or feedback:

📺 YouTube: [@AliToori](https://youtube.com/@AliToori)  
💬 Telegram: [@AliToori](https://t.me/@AliToori)  
📂 GitHub: [github.com/AliToori](https://github.com/AliToori)