import kagglehub
import os
import pandas as pd
import numpy as np
import re

# using the public enron-spam dataset to train, which is a labeled subset of the enron email dataset
path = kagglehub.dataset_download("wanderfj/enron-spam")

print("Path to dataset files:", path)

root_path = path
emails = []
labels = []

for folder in ['enron1', 'enron2', 'enron3', 'enron4', 'enron5', 'enron6']:
    ham_path = os.path.join(root_path, folder, 'ham')
    spam_path = os.path.join(root_path, folder, 'spam')
    
    # ham emails
    for filename in os.listdir(ham_path):
        file_path = os.path.join(ham_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            emails.append(content)
            labels.append(0)  # Label 0 for non-spam)
    
    # spam emails
    for filename in os.listdir(spam_path):
        file_path = os.path.join(spam_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            emails.append(content)
            labels.append(1)  # Label 1 for spam

email_df = pd.DataFrame({'email_text': emails, 'label': labels})

def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# load and preprocess data
data=email_df
data['combined_text'] = data['email_text'].apply(preprocess_text)

# Additional numeric feature extraction on unprocessed dataset
data['word_count'] = data['email_text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
data['char_count'] = data['email_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
data['avg_word_length'] = data['email_text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if isinstance(x, str) and x.split() else 0)
data['exclamation_count'] = data['email_text'].apply(lambda x: x.count('!') if isinstance(x, str) else 0)
data['dollar_sign_count'] = data['email_text'].apply(lambda x: x.count('$') if isinstance(x, str) else 0)
data['uppercase_word_count'] = data['email_text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()) if isinstance(x, str) else 0)
data['uppercase_char_percentage'] = data['email_text'].apply(lambda x: sum(1 for char in x if char.isupper()) / len(x) if isinstance(x, str) and len(x) > 0 else 0)

train_data = data

# naive bayes classifier for mixed features
class MixedNaiveBayesClassifier:
    def __init__(self):
        self.prior_probs = {}
        self.likelihoods = {}
        self.gaussian_params = {}
        self.vocab = set()
    
    def fit(self, X_text, X_numeric, y):
        # calculate prior probabilities
        total_samples = len(y)
        for label in np.unique(y):
            self.prior_probs[label] = np.sum(y == label) / total_samples
            
            # text features for each class
            texts = X_text[y == label]
            word_counts = {}
            for text in texts:
                words = text.split()
                for word in words:
                    self.vocab.add(word)
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
            self.likelihoods[label] = word_counts
            
            # process numeric features (Gaussian)
            numeric_data = X_numeric[y == label]
            self.gaussian_params[label] = {
                feature: (numeric_data[feature].mean(), numeric_data[feature].std() + 1e-6) 
                for feature in numeric_data.columns
            }
    
    def gaussian_likelihood(self, x, mean, std):
        """Compute Gaussian probability density function."""
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X_text, X_numeric):
        predictions = []
        for text, numeric_values in zip(X_text, X_numeric.iterrows()):
            _, numeric_values = numeric_values
            posteriors = {}
            for label in self.prior_probs:
                # Start with the prior probability
                posterior = np.log(self.prior_probs[label])
                
                # Process text part
                words = text.split()
                for word in words:
                    word_likelihood = (self.likelihoods[label].get(word, 0) + 1) / (sum(self.likelihoods[label].values()) + len(self.vocab))
                    posterior += np.log(word_likelihood)
                
                # Process numeric part using Gaussian distribution for each numeric feature
                for feature, value in numeric_values.items():
                    mean, std = self.gaussian_params[label][feature]
                    posterior += np.log(self.gaussian_likelihood(value, mean, std))
                
                posteriors[label] = posterior
            
            # choosing class with the highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        
        return predictions

X_text_train = train_data['combined_text']
X_numeric_train = train_data[['word_count', 'char_count', 'avg_word_length', 
                              'exclamation_count', 'dollar_sign_count', 'uppercase_word_count', 'uppercase_char_percentage']]
y_train = train_data['label']

# Train model
nb_classifier = MixedNaiveBayesClassifier()
nb_classifier.fit(X_text_train, X_numeric_train, y_train)

#-----------------------------------------------------------------------------------------
# Processing the test data and predicting on it

test_folder = 'test'  # Name of the folder containing test emails
test_emails=[]

for filename in os.listdir(test_folder):
        if filename.startswith('email') and filename.endswith('.txt'):
            file_path = os.path.join(test_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                test_emails.append(content)

test_email_df = pd.DataFrame({'email_text': test_emails})
data=test_email_df
data['combined_text'] = data['email_text'].apply(preprocess_text)

# numeric feature extraction
data['word_count'] = data['email_text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
data['char_count'] = data['email_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
data['avg_word_length'] = data['email_text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if isinstance(x, str) and x.split() else 0)
data['exclamation_count'] = data['email_text'].apply(lambda x: x.count('!') if isinstance(x, str) else 0)
data['dollar_sign_count'] = data['email_text'].apply(lambda x: x.count('$') if isinstance(x, str) else 0)
data['uppercase_word_count'] = data['email_text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()) if isinstance(x, str) else 0)
data['uppercase_char_percentage'] = data['email_text'].apply(lambda x: sum(1 for char in x if char.isupper()) / len(x) if isinstance(x, str) and len(x) > 0 else 0)

test_data = data
            
X_text_test = test_data['combined_text']
X_numeric_test = test_data[['word_count', 'char_count', 'avg_word_length', 
                            'exclamation_count', 'dollar_sign_count', 'uppercase_word_count', 'uppercase_char_percentage']]

# predictions on the test set
test_predictions = nb_classifier.predict(X_text_test, X_numeric_test)

predictions = [int(pred) for pred in test_predictions]
print(f'predictions: {predictions}')

# # Evaluate accuracy
# y_test = <pls define>
# accuracy = np.mean(test_predictions == y_test)
# print(f'Mixed Naive Bayes Classifier Accuracy: {accuracy:.2f}')
 