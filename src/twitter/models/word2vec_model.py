# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from preprocessing.tokenizer from Tokenizer
from models.word2vec import Word2VecAvarager


def build_model(n_dim=25):
    w2v_filename = "data/glove.twitter.27B/word2vec.twitter.27B.{}d.txt".format(n_dim)

    clf_class = ExtraTreesClassifier
    # clf_class = MLPClassifier((30, 30))

    pipe_params = {
        # 'clf__n_jobs': -1,
        # 'clf__n_estimators': 300,
        # 'clf__class_weight': 'balanced',
        # 'clf__random_state': 42,
    }

    pipeline = Pipeline([
        ('tknz', Tokenizer()),
        ('w2v', Word2VecAvarager(w2v_filename)),
        ('clf', clf_class),
    ])

    pipeline.set_params(**pipe_params)

    return pipeline
