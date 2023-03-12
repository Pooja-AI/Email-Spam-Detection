import logging
import traceback
import pickle
import email
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn import svm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# Set up logging
logging.basicConfig(filename='log/spam_classification.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Load the path for each email file for both categories.
try:
    logging.info('Loading email file paths')
    ham_files = train_test_split(glob.glob('./data/20030228_hard_ham/hard_ham/*'), random_state=123)
    spam_files = train_test_split(glob.glob('./data/20050311_spam_2/spam_2/*'), random_state=123)
except Exception as e:
    logging.error("An error occurred while loading email file paths: %s", e)
    logging.error(traceback.format_exc())

# Method for removing the stop words.
def remove_stop_words(input):
    result = [i for i in input if i not in ENGLISH_STOP_WORDS]
    return result

# Create the lemmatizer.
lemmatizer = WordNetLemmatizer()

# Method for lemmatizing the text.
def lemmatize_text(input):
    return [lemmatizer.lemmatize(i) for i in input]

# Method for getting the content of an email.
def get_content(filepath):
    try:
        file = open(filepath, encoding='latin1')
        message = email.message_from_file(file)

        for msg_part in message.walk():
            # Keep only messages with text/plain content.
            if msg_part.get_content_type() == 'text/plain':
                return msg_part.get_payload()
    except Exception as e:
        logging.error("An error occurred while getting email content from file '%s': %s", filepath, e)
        logging.error(traceback.format_exc())
def train_test_data():
    try:
        # Get the training and testing data.
        logging.info('Loading training and testing data')
        ham_train_data = [get_content(i) for i in ham_files[0]]
        ham_test_data = [get_content(i) for i in ham_files[1]]
        spam_train_data = [get_content(i) for i in spam_files[0]]
        spam_test_data = [get_content(i) for i in spam_files[1]]

        # Keep emails with non-empty content.
        ham_train_data = list(filter(None, ham_train_data))
        ham_test_data = list(filter(None, ham_test_data))
        spam_train_data = list(filter(None, spam_train_data))
        spam_test_data = list(filter(None, spam_test_data))

        # Merge the train/test files for both categories.
        train_data = np.concatenate((ham_train_data, spam_train_data))
        test_data = np.concatenate((ham_test_data, spam_test_data))

        # Assign a class for each email (ham = 0, spam = 1).
        ham_train_class = [0]*len(ham_train_data)
        ham_test_class = [0]*len(ham_test_data)
        spam_train_class = [1]*len(spam_train_data)
        spam_test_class = [1]*len(spam_test_data)

        # Merge the train/test classes for both categories.
        train_class = np.concatenate((ham_train_class, spam_train_class))
        test_class = np.concatenate((ham_test_class, spam_test_class))

        # Tokenize the train/test data.
        logging.info('Tokenizing the data')
        train_data = [word_tokenize(i) for i in train_data]
        test_data = [word_tokenize(i) for i in test_data]

        # Remove the stop words.
        logging.info('Removing stop words')
        train_data = [remove_stop_words(i) for i in train_data]
        test_data = [remove_stop_words(i) for i in test_data]

        # Lemmatize the text.
        logging.info('Lemmatizing the text')
        train_data = [lemmatize_text(i) for i in train_data]
        test_data = [lemmatize_text(i) for i in test_data]

        # Reconstruct the data.
        logging.info('Reconstruct the data.')
        train_data = [" ".join(i) for i in train_data]
        test_data = [" ".join(i) for i in test_data]
    except Exception as e:
        logging.error("Error occurred while loading and processing the data.")
        logging.error(str(e))
        raise

    return train_data, test_data, train_class, test_class


try:
    # Load the data.
    logging.info("Loading data...")
    train_data, test_data, train_class, test_class = train_test_data()

    # Save the data to a file.
    logging.info("Saving data...")
    with open('models/train_test_data.pkl', 'wb') as f:
        pickle.dump((train_data, test_data, train_class, test_class), f)

    # Create the vectorizer.
    logging.info("Creating vectorizer...")
    vectorizer = TfidfVectorizer()

    # Fit with the train data.
    logging.info("Fitting vectorizer with train data...")
    vectorizer.fit(train_data)

    # Transform the test/train data into features.
    logging.info("Transforming data into features...")
    train_data_features = vectorizer.transform(train_data)
    test_data_features = vectorizer.transform(test_data)

    # Create the classifier.
    logging.info("Creating classifier...")
    svm_classifier = svm.SVC(kernel="rbf", C=1.0, gamma=1.0, probability=True)

    # Fit the classifier with the train data.
    logging.info("Fitting classifier with train data...")
    svm_classifier.fit(train_data_features.toarray(), train_class)

    # Save the classifier to a file.
    logging.info("Saving classifier...")
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)

    # Get the classification score of the train data.
    logging.info("Calculating classification score of train data...")
    train_score = svm_classifier.score(train_data_features.toarray(), train_class)
    logging.info("Train score: %s", train_score)

    # Get the classification score of the test data.
    logging.info("Calculating classification score of test data...")
    test_score = svm_classifier.score(test_data_features.toarray(), test_class)
    logging.info("Test score: %s", test_score)

except Exception as e:
    logging.error("An error occurred: %s", str(e))
