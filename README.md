# SMS Spam Classifier

A machine learning project in Python that classifies SMS messages as **spam** or **not spam**.
It compares two models, Naive Bayes and Logistic Regression, on TF-IDF text features and
saves the better one for predictions.

## Dataset
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection):
5,574 labelled SMS messages (4,827 ham, 747 spam). The script downloads it automatically on first run.
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
Evaluated on a held-out test set of 1,115 messages (20% split, `random_state=42`):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 97.04% | 100% | 77.85% | 87.55% |
| Logistic Regression | 96.77% | 100% | 75.84% | 86.26% |

**Key takeaways**
- Both models reach about 97% accuracy.
- Precision is 100%: no normal message was wrongly marked as spam, which matters because blocking a real message is costly.
- Recall is about 76-78%: roughly 1 in 5 spam messages still slips through. The dataset is imbalanced (about 13% spam), so accuracy alone can be misleading.
- Naive Bayes had the best F1 score, so it is the model saved and used for predictions.

**Example**
```
$ python spam_classifier.py "Congratulations! You won a free prize, click now"
spam
$ python spam_classifier.py "Hey, are we still meeting for lunch tomorrow?"
not spam
```

## Tech stack
Python, pandas, scikit-learn, joblib

## Future improvements
- Try an SVM or a neural network
- Build a small web interface with Flask or Streamlit
- Improve recall on the imbalanced data (class weights, resampling, or threshold tuning)

## Author
Anshuman Rai: [GitHub](https://github.com/rai7anshu-commits)
