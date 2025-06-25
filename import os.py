import tarfile
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

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

# Paths to tar files
paths = {
    "spam": "C:/Users/20021010_spam.tar",
    "easy_ham": "C:/Users/20021010_easy_ham.tar",
    "hard_ham": "C:/Users/20021010_hard_ham.tar"
}

# Extract tar files to temporary folders
extracted_paths = {}
for label, path in paths.items():
    extract_dir = f"C:/Users/mmehr/OneDrive/Desktop/CHALMERS/3_ThirdYear/3_StudyPeriod2/Introduction to data science and AI - DAT565/Assignments/Assignment_3/{label}_emails"
    with tarfile.open(path) as tar:
        tar.extractall(path=extract_dir)
    extracted_paths[label] = extract_dir

# Function to preprocess and tokenize email content manually
def preprocess_email(email_content):
    # Remove punctuation and convert to lowercase
    email_content = re.sub(r'\W+', ' ', email_content).lower()
    # Tokenize by splitting on spaces
    words = email_content.split()
    # Remove stopwords
    filtered_words = [word for word in words if word not in manual_stopwords]
    return filtered_words

# Function to read emails and count word frequencies
def analyze_emails(folder_path):
    word_counter = Counter()
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, encoding='utf-8', errors='ignore') as email:
                content = email.read()
                words = preprocess_email(content)
                word_counter.update(words)
    return word_counter.most_common(10)

# Analyze each dataset
results = {}
for label, path in extracted_paths.items():
    results[label] = analyze_emails(path)

# Print results
#for label, common_words in results.items():
   # print(f"\nMost common words in {label}:")
    #for word, count in common_words:
      #  print(f"{word}: {count}")


####################################################################part B


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
            labels.append(1)  # Spam label

# Easy Ham emails (label 0)
for root, _, files in os.walk(extracted_paths["easy_ham"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            content = email.read()
            emails.append(preprocess_email(content))
            labels.append(0)  # Ham label

# Hard Ham emails (label 0)
for root, _, files in os.walk(extracted_paths["hard_ham"]):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        with open(file_path, encoding='utf-8', errors='ignore') as email:
            content = email.read()
            emails.append(preprocess_email(content))
            labels.append(0)  # Ham label

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.3, random_state=42)

# Convert emails to bag-of-words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the classifier
#print("Classification Report:")
#print(classification_report(y_test, y_pred))

def preprocess_email(email_content):
    # Remove URLs
    email_content = re.sub(r'http\S+', '', email_content)
    # Remove HTML tags
    email_content = re.sub(r'<.*?>', '', email_content)
    # Remove punctuation and convert to lowercase
    email_content = re.sub(r'\W+', ' ', email_content).lower()
    # Tokenize by splitting on spaces
    words = email_content.split()
    # Remove stopwords
    filtered_words = [word for word in words if word not in manual_stopwords]
    return ' '.join(filtered_words)  # Return as a single string for vectorizer




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
print("\nFirst 10 words in the vocabulary:", feature_names[:20])

# You can also explore the most frequent words in the entire dataset
word_count = X.sum(axis=0).A1  # Sum across the rows to get word counts
word_count_dict = dict(zip(feature_names, word_count))
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)

# Print top 10 most frequent words
print("\nTop 10 most frequent words in the entire dataset:")
for word, count in sorted_word_count[:20]:
    print(f"{word}: {count}")


    # Define the list of specific spam-related words
    spam_related_words = [
    "win", "money", "cash", "prize", "offer", "deal", "earn", "profit", 
    "income", "bonus", "reward", "payout", "jackpot", "fund", "investment", 
    "hurry", "urgent", "immediate", "limited", "now", "today", "act", "last", 
    "expires", "final", "don't miss", "free", "gift", "giveaway", "trial", 
    "access", "exclusive", "unlimited", "guarantee", "promise", "best", 
    "amazing", "incredible", "once-in-a-lifetime", "no risk", "proven", 
    "100%", "discount", "sale", "cheap", "bargain", "clearance", "save", 
    "affordable", "click", "confirm", "password", "bank", "account", 
    "verify", "security", "update", "credentials", "login", "congratulations", 
    "winner", "selected", "luxury", "vacation", "lottery", "sweepstakes", 
    "insurance", "weight loss", "cure", "treatment", "anti-aging", 
    "medication", "pills", "supplement"
]


# Function to count specific words' frequencies in a dataset
def count_specific_words(folder_path, specific_words):
    word_counter = Counter({word: 0 for word in specific_words})  # Initialize counter for specific words
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, encoding='utf-8', errors='ignore') as email:
                content = email.read()
                words = preprocess_email(content).split()  # Tokenize preprocessed content
                for word in specific_words:
                    word_counter[word] += words.count(word)
    return word_counter

# Analyze specific words for each dataset
specific_word_results = {}
for label, path in extracted_paths.items():
    specific_word_results[label] = count_specific_words(path, spam_related_words)

# Print results
for label, word_counts in specific_word_results.items():
    print(f"\nSpecific word frequencies in {label}:")
    for word, count in word_counts.items():
        print(f"{word}: {count}")

