from nltk.corpus import stopwords
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from preprocessing.text_cleaner import TextCleaner

from tools.memorize_decorator import MemoDecorator as MD


def build_model():
    clf = ExtraTreesClassifier()

    stops = set(stopwords.words("english"))

    pipe_params = {
        'cleantxt__memorize_fit': False,
        'vect__memorize_fit': False,
        'vect__max_df': .7,
        'vect__max_features': 10000,
        'vect__ngram_range': (1, 2),
        'vect__strip_accents': 'unicode',
        'vect__stop_words': stops,
        'tfidf__norm': 'l2',
        'tfidf__use_idf': True,
        'clf__n_jobs': -1,
        'clf__n_estimators': 300,
        'clf__class_weight': 'balanced',
        'clf__random_state': 42,
    }

    pipeline = Pipeline([
        ('cleantxt', MD(TextCleaner())),
        ('vect', MD(CountVectorizer())),
        ('tfidf', MD(TfidfTransformer())),
        ('clf', MD(clf)),
    ])

    pipeline.set_params(**pipe_params)

    return pipeline
