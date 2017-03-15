import logging

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline


def build_model(n_dim=25):
    w2v_filename = "data/glove.twitter.27B/word2vec.twitter.27B.{}d.txt".format(n_dim)

    classifier = ExtraTreesClassifier()

    # choosing w2v model dimensions (25, 50, 100, 200)
    n_dim = 100
    w2v_filename = "data/glove.twitter.27B/word2vec.twitter.27B.{}d.txt".format(n_dim)
    from models.word2vec_model import Word2VecAvarager
    s2a = Word2VecAvarager(w2v_filename)
    X = s2a.fit_transform(X)

    # stops = set(stopwords.words("english"))

    pipe_params = {
        'transformer__filename': w2v_filename,
        'clf__class_weight': 'balanced',
        'clf__n_estimators': 300,
        'clf__random_state': 42,
    }

    pipeline = Pipeline([
        ('clf', classifier),
    ])

    logging.info("built model")

    return MLPClassifier(hidden_layer_sizes=(30, 30))
