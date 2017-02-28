from nltk.corpus import stopwords
from preprocessing import TextCleaner
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


def build_model(classifier):
    stops = set(stopwords.words("english"))

    pipe_params = {
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
        ('clf', classifier),
    ])

    pipeline.set_params(**pipe_params)

    return pipeline


def build_simple_model():
    classifier = ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
    return build_model(classifier)
