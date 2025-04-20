# 📰 True and Fake News Classification

This project focuses on the classification of news articles as either "true" or "fake" using machine learning techniques. By training models on a labeled news dataset, the goal is to predict the authenticity of news articles based on their textual content.

## 📊 Features

- ✅ Text preprocessing (tokenization, stopword removal, etc.)
- ✅ Natural Language Processing (NLP) techniques
- ✅ Various machine learning models for text classification (e.g., Logistic Regression, Random Forest, etc.)
- ✅ Deep learning models for improved accuracy (e.g., LSTM, BERT)
- ✅ Evaluation using accuracy, precision, recall, and F1-score

## 📁 Dataset

The dataset contains news articles labeled as "true" or "fake." It includes various features such as:

- **Title**: The headline of the news article
- **Text**: The body/content of the article
- **Label**: The authenticity of the article (True/False)

> _Note: You can find the dataset in the `news_data.csv` file or use a popular dataset like the [Fake News Dataset on Kaggle](https://www.kaggle.com/c/fake-news/data)._

## 🚀 How to Run

1. **Clone the repo:**
```bash
git clone https://github.com/YOUR-USERNAME/true-fake-news-dataset.git
cd true-fake-news-dataset

    Install dependencies:

pip install -r requirements.txt

    Run the main script:

python news_classification.py

    Results:

    A trained machine learning model will be saved in the model/ directory

    Evaluation metrics (accuracy, precision, recall) will be printed to the console

📦 Output

    model/: Folder containing the saved trained models

    predictions.csv: File containing the model’s predictions for test data

    Evaluation metrics displayed in the terminal

📈 Sample Output

Accuracy: 0.95
Precision: 0.94
Recall: 0.93
F1-Score: 0.94

🧠 Techniques Used

    Text Preprocessing: Tokenization, stopword removal, lemmatization

    Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency)

    Machine Learning Models: Logistic Regression, Random Forest, Naive Bayes

    Deep Learning Models: LSTM (Long Short-Term Memory), BERT (Bidirectional Encoder Representations from Transformers)

🛠️ Requirements

    pandas

    numpy

    scikit-learn

    nltk (for NLP processing)

    tensorflow (for deep learning models)

You can install all dependencies with:

pip install pandas numpy scikit-learn nltk tensorflow

🤝 Contributing

Feel free to fork the repo, open issues, or submit pull requests to improve the project. Contributions are welcome!
📄 License

This project is licensed under the MIT License.
📬 Contact

Made with ❤️ by Aman Chaurasia
