from nltk.corpus import stopwords
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from preprocessing.text_cleaner import TextCleaner


def build_model():
    clf_class = ExtraTreesClassifier

    stops = set(stopwords.words("english"))

    pipe_params = {
        'clf__n_jobs': -1,
        'clf__n_estimators': 300,
        'clf__class_weight': 'balanced',
        'clf__random_state': 42,
        'tfidf__norm': 'l2',
        'tfidf__use_idf': True,
        'vect__max_df': .7,
        'vect__max_features': 10000,
        'vect__ngram_range': (1, 2),
        'vect__strip_accents': 'unicode',
        'vect__stop_words': stops
    }

    pipeline = Pipeline([
        ('cleantxt', TextCleaner()),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', clf_class()),
    ])

    pipeline.set_params(**pipe_params)

    return pipeline
