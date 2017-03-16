from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec

from preprocessing.replacer import Replacer
from preprocessing.tweet_tokenizer import TweetTokenizerTransformer

from models.word2vec import Word2VecAvarager


def build_model(n_dim=25):
    w2v_filename = ("data/glove.twitter.27B/"
                    "word2vec.twitter.27B.{}d.txt".format(n_dim))

    # clf = ExtraTreesClassifier()
    clf = MLPClassifier()
    # clf = LogisticRegression()

    # loading w2v once to avoid multiple loads
    wv = Word2Vec.load_word2vec_format(w2v_filename)
    wv.init_sims(replace=True)

    pipe_params = {
        'w2v__filename': w2v_filename,
        'w2v__wv': wv,
        # 'clf__n_jobs': -1,
        # 'clf__n_estimators': 300,
        # 'clf__class_weight': 'balanced',
        # 'clf__random_state': 42,
        'clf__hidden_layer_sizes': (10, ),
    }

    # loading here in order not to load over and over

    pipeline = Pipeline([
        ('replace', Replacer()),
        ('tokenize', TweetTokenizerTransformer()),
        ('w2v', Word2VecAvarager()),
        ('clf', clf),
    ])

    pipeline.set_params(**pipe_params)

    return pipeline
