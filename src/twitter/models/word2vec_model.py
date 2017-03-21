from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

from preprocessing.replacer import Replacer
from preprocessing.tweet_tokenizer import TweetTokenizerTransformer

from models.word2vec import Word2VecAvarager

from tools.memorize_decorator import MemoDecorator as MD


def build_model(n_dim=25):
    w2v_filename = ("data/glove.twitter.27B/"
                    "word2vec.twitter.27B.{}d.txt".format(n_dim))

    # clf = ExtraTreesClassifier()
    clf = MLPClassifier()
    # clf = LogisticRegression()

    pipe_params = {
        'tokenize__memorize_fit': False,
        'replace__memorize_fit': False,
        'w2v__filename': w2v_filename,
        'w2v__memorize_fit': False,
        # 'w2v__wv': wv,
        # 'clf__n_jobs': -1,
        # 'clf__n_estimators': 300,
        # 'clf__class_weight': 'balanced',
        # 'scale': ,
        'clf__random_state': 42,
        'clf__hidden_layer_sizes': (40, 40),
    }

    pipeline = Pipeline([
        ('replace', MD(Replacer())),
        ('tokenize', MD(TweetTokenizerTransformer())),
        ('w2v', MD(Word2VecAvarager())),
        # ('scale', MD(StandardScaler())),
        ('clf', MD(clf)),
    ])

    pipeline.set_params(**pipe_params)

    return pipeline
