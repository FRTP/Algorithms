from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from preprocessing.text_cleaner import TextCleaner

from tools.memorize_decorator import MemoDecorator as MD

from models.baseline_model import build_model as baseline
from models.word2vec_model import build_model as w2v


def build_model():
    models = [
        ('baseline', baseline()),
        ('w2v', w2v()),
    ]

    for i, _ in enumerate(models):
        models[i][1].steps.pop()

    union = FeatureUnion(models)
    clf = ExtraTreesClassifier()

    pipeline = Pipeline([
        ('union', union),
        ('clf', MD(clf)),
    ])

    pipe_params = {
        'clf__n_jobs': -1,
        'clf__n_estimators': 300,
        'clf__class_weight': 'balanced',
        'clf__random_state': 42,
        # 'clf__hidden_layer_sizes': (40, 40),
    }

    pipeline.set_params(**pipe_params)

    return pipeline
