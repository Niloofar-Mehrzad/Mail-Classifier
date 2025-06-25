import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

# Define the paths to the extracted email folders
extracted_paths = {
    "spam": "C:/Users/spam_emails",
    "easy_ham": "C:/easy_ham_emails",
    "hard_ham": "C:/hard_ham_emails"
}

# Hardcoded list of common stopwords
manual_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now"
])

# Function to preprocess and tokenize email content
def preprocess_email(email_content):
    # Remove punctuation and convert to lowercase
    email_content = re.sub(r'\W+', ' ', email_content).lower()
    # Tokenize by splitting on spaces
    words = email_content.split()
    # Remove stopwords
    filtered_words = [word for word in words if word not in manual_stopwords]
    return ' '.join(filtered_words)  # Return as a single string for vectorizer

# Load spam, easy ham, and hard ham emails
spam_emails = []
easy_ham_emails = []
hard_ham_emails = []

# Load emails for each category
for root, _, files in os.walk(extracted_paths["spam"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            spam_emails.append(preprocess_email(email.read()))

for root, _, files in os.walk(extracted_paths["hard_ham"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            hard_ham_emails.append(preprocess_email(email.read()))

# Labels for spam and ham emails
spam_labels = [1] * len(spam_emails)
hard_ham_labels = [0] * len(hard_ham_emails)

# Prepare training and test datasets
train_emails = spam_emails + hard_ham_emails
train_labels = spam_labels + hard_ham_labels

# Use hard ham and a portion of spam for testing
test_emails = hard_ham_emails + spam_emails[:len(hard_ham_emails)]  # Balanced testing dataset
test_labels = hard_ham_labels + spam_labels[:len(hard_ham_emails)]

# Vectorize the training and test datasets
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_emails)
X_test = vectorizer.transform(test_emails)

# Train Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb.fit(X_train, train_labels)
y_pred_mnb = mnb.predict(X_test)

# Train Bernoulli Naive Bayes Classifier
bnb = BernoulliNB()
bnb.fit(X_train, train_labels)
y_pred_bnb = bnb.predict(X_test)

# Evaluate and print metrics for Multinomial Naive Bayes
print("Multinomial Naive Bayes Classification Report:")
print(classification_report(test_labels, y_pred_mnb))
print("\nConfusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(test_labels, y_pred_mnb))

# Evaluate and print metrics for Bernoulli Naive Bayes
print("\nBernoulli Naive Bayes Classification Report:")
print(classification_report(test_labels, y_pred_bnb))
print("\nConfusion Matrix for Bernoulli Naive Bayes:")
print(confusion_matrix(test_labels, y_pred_bnb))
