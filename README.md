# SMS Spam Classifier

A machine learning project in Python that classifies SMS messages as **spam** or **not spam**.
It compares two models, Naive Bayes and Logistic Regression, on TF-IDF text features and
saves the better one for predictions.

## Dataset
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection):
5,572 labelled SMS messages. The script downloads it automatically on first run.
(If the download fails, get the zip from the link, extract the `SMSSpamCollection` file,
and put it in this folder.)

## How it works
1. Load and label the data (spam = 1, ham = 0).
2. Split 80% train / 20% test (stratified).
3. Convert text to TF-IDF features (lowercase, English stop words removed).
4. Train Naive Bayes and Logistic Regression, then compare accuracy, precision, recall and F1.
5. Save the best model and use it to classify new messages.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python spam_classifier.py train
python spam_classifier.py "Congratulations! You won a free prize, click now"
```

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | | | | |
| Logistic Regression | | | | |

## Tech stack
Python, pandas, scikit-learn, Joblib 

## Author
Anshuman Rai: [GitHub](https://github.com/rai7anshu-commits)
