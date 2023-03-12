import logging
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# Configure the logging module.
logging.basicConfig(filename='spam_classification.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load the preprocessed train and test data
logging.info("Loading train and test data.")
with open('models/train_test_data.pkl', 'rb') as f:
    train_data, test_data, train_class, test_class = pickle.load(f)

# Load the saved SVM model.
logging.info("Loading SVM model.")
with open('models/svm_model.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)
def prediction_model(unseen_data):
    # Preprocess the unseen data.
    logging.info("Preprocessing unseen data.")
    unseen_data_original = unseen_data # Save the original text for later.
    unseen_data = word_tokenize(unseen_data)
    unseen_data = [i for i in unseen_data if i not in ENGLISH_STOP_WORDS]
    lemmatizer = WordNetLemmatizer()
    unseen_data = [lemmatizer.lemmatize(i) for i in unseen_data]
    unseen_data = " ".join(unseen_data)

    # Transform the preprocessed unseen data into features.
    logging.info("Transforming unseen data into features.")
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(train_data)
    unseen_data_features = vectorizer.transform([unseen_data]).toarray()
    # Get the words with the highest TF-IDF scores.
    max_tf_idf_scores = np.asarray(unseen_data_features.max(axis=0)).reshape(-1)
    sorted_indexes = max_tf_idf_scores.argsort()[::-1]

    # Get the feature names (words) and their corresponding scores.
    feature_names = np.array(vectorizer.get_feature_names_out())
    feature_scores = np.array(unseen_data_features[0])

    # Print the top 10 spam words and their scores.
    logging.info("Printing top spam words.")
    n_top_words = 10
    top_word_indices = sorted_indexes[:n_top_words]
    for i, word_index in enumerate(top_word_indices):
        word = feature_names[word_index]
        score = feature_scores[word_index]
        logging.info(f"{i+1}. {word}: {score:.3f}")

    # Highlight the spam words in the original text.
    logging.info("Highlighting spam words in original text.")
    spam_word_indices = np.where(feature_scores > 0)[0]
    start_tag = "\033[1;31m" # Red color for spam words.
    end_tag = "\033[m" # Reset color to default.
    highlighted_text = unseen_data_original
    list_data = []
    for index in spam_word_indices:
        word = feature_names[index]
        list_data.append(word)
        # highlighted_text = highlighted_text.replace(word, f"{start_tag}{word}{end_tag}")

    logging.info("Printing predicted label, original text, and highlighted text.")
    predicted_label = svm_classifier.predict(unseen_data_features)[0]
    if predicted_label == 1:
        predict = "SPAM "
        highlighted_words = list_data
    else:
        predict = "HAM"
        highlighted_words = list_data
    
    return unseen_data_original,predict,highlighted_words

