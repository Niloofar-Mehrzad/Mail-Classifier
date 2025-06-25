import os
import re
import tarfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

# Define the paths to the extracted email folders
extracted_paths = {
    "spam": "C:/Users/spam_emails",
    "easy_ham": "C:/Users/easy_ham_emails",
    "hard_ham": "C:/Users/hard_ham_emails"
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

# Load all emails and assign labels
emails = []
labels = []

# Spam emails (label 1)
for root, _, files in os.walk(extracted_paths["spam"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            content = email.read()
            emails.append(preprocess_email(content))
            labels.append(1)  # Label for spam

# Easy Ham emails (label 0)
for root, _, files in os.walk(extracted_paths["hard_ham"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            content = email.read()
            emails.append(preprocess_email(content))
            labels.append(0)  # Label for easy ham


# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the emails into a document-term matrix (DTM)
X = vectorizer.fit_transform(emails)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

# Train Bernoulli Naive Bayes Classifier
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

# Evaluate and print metrics
print("Multinomial Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_mnb))

print("\nBernoulli Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_bnb))

# Confusion Matrices
print("\nConfusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test, y_pred_mnb))

print("\nConfusion Matrix for Bernoulli Naive Bayes:")
print(confusion_matrix(y_test, y_pred_bnb))
