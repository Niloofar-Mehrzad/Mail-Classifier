import os
import re
import tarfile
from sklearn.feature_extraction.text import CountVectorizer

# Define the paths to the extracted email folders
extracted_paths = {
    "spam": "C:/Users/spam_emails",
    "easy_ham": "C:/Users//easy_ham_emails",
    "hard_ham": "C:/Users//hard_ham_emails"
}

# Define the tar archive paths
tar_files = {
    "spam": "C:/Users/20021010_spam.tar.bz2",
    "easy_ham": "C:/20021010_easy_ham.tar.bz2",
    "hard_ham": "C:/Users/20021010_hard_ham.tar.bz2"
}

# Function to extract tar files without filtering (using filter=None)
def extract_tar_file(tar_path, extract_dir):
    with tarfile.open(tar_path, 'r:bz2') as tar:
        tar.extractall(path=extract_dir, filter=None)

# Extract all tar archives
for category, tar_path in tar_files.items():
    extract_dir = extracted_paths[category]
    os.makedirs(extract_dir, exist_ok=True)  # Ensure the directory exists
    print(f"Extracting {category} to {extract_dir}...")
    extract_tar_file(tar_path, extract_dir)

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
for root, _, files in os.walk(extracted_paths["easy_ham"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            content = email.read()
            emails.append(preprocess_email(content))
            labels.append(0)  # Label for easy ham

# Hard Ham emails (label 0)
for root, _, files in os.walk(extracted_paths["hard_ham"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            content = email.read()
            emails.append(preprocess_email(content))
            labels.append(0)  # Label for hard ham

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the emails into a document-term matrix (DTM)
X = vectorizer.fit_transform(emails)

# Get feature names (words in the vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Convert the matrix to a dense format and print the shape of the document-term matrix
X_dense = X.toarray()
print("Shape of the document-term matrix:", X_dense.shape)

# Print the first 10 words in the vocabulary
print("\nFirst 10 words in the vocabulary:", feature_names[:10])

# You can also explore the most frequent words in the entire dataset
word_count = X.sum(axis=0).A1  # Sum across the rows to get word counts
word_count_dict = dict(zip(feature_names, word_count))
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)

# Print top 10 most frequent words
print("\nTop 10 most frequent words in the entire dataset:")
for word, count in sorted_word_count[:10]:
    print(f"{word}: {count}")
