# Healthcare Costs Prediction using TensorFlow Linear Regression

This project implements a healthcare costs prediction system using a Linear Regression model in TensorFlow, developed as part of the freeCodeCamp Machine Learning with Python certification. The goal is to predict healthcare expenses based on features like age, sex, BMI, children, smoker status, and region, achieving a Mean Absolute Error (MAE) under $3500 on the test dataset, using the insurance dataset.

---

👨‍💻 **Author**: Ali Toori – Full-Stack Python Developer  
📺 **YouTube**: [@AliToori](https://youtube.com/@AliToori)  
💬 **Telegram**: [@AliToori](https://t.me/@AliToori)  
📂 **GitHub**: [Github.com/AliToori](https://github.com/AliToori)

---

### Project Overview
The project involves:
1. Loading and preprocessing the insurance dataset, encoding categorical variables (sex, smoker, region), and normalizing numerical features.
2. Splitting the dataset into 80% training and 20% testing sets, with the target variable (`expenses`) separated as labels.
3. Building a Linear Regression model using TensorFlow’s `Sequential` API with a single dense layer.
4. Training the model to minimize MAE and evaluating it to ensure MAE < 3500 on the test set.
5. Visualizing predicted vs. actual expenses using a scatter plot.
6. Implementing the logic in a modular, class-based Python script (`HealthcareCostsPredictor.py`) for reusability.

Example output from the evaluation:
```bash
Testing set Mean Abs Error: 2456.78 expenses
You passed the challenge. Great job!
```
(A scatter plot shows predicted vs. actual expenses with a 1:1 reference line.)

---

### [Google Colab Project Link](https://colab.research.google.com/drive/1YhqiUuH22rZCzQpfbL8msT8cHZ4J_uGR#scrollTo=Xe7RXH3N3CWU)

---

## 🛠 Tech Stack
* Language: Python 3.8+
* Libraries:
  * TensorFlow (for Linear Regression model with `Sequential` API)
  * Pandas (for data preprocessing and manipulation)
  * NumPy (for numerical operations)
  * Scikit-learn (for `LabelEncoder`, `StandardScaler`, and `train_test_split`)
  * Matplotlib (for visualization)
* Tools:
  * Google Colab for development, training, and testing (with GPU support)
  * GitHub for version control

---

## 📂 Project Structure
The project includes:
* `HealthcareCostsPredictor.py`: A class-based Python script with methods for loading, preprocessing, building, training, evaluating, and visualizing the model.
* `insurance.csv`: The dataset (downloaded automatically from [https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv)).
* Colab Notebook (optional): A four-cell notebook implementing the same logic sequentially (import libraries, load data, preprocess/train, evaluate/visualize).
* `README.md`: This file.

Dataset structure:
```bash
insurance.csv: Contains features (age, sex, bmi, children, smoker, region) and target (expenses)
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