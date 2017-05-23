from src.twitter.models import baseline_model
from src.twitter.models import word2vec_model

import pandas as pd

model_builder = {
    'baseline': baseline_model.build_model,
    'word2vec': word2vec_model.build_model,
}


def generate_features(tweets=None, data_file=None, model="baseline"):
    if isinstance(model, str):
        model = model_builder[model]()
    if tweets is None:
        if data_file is None:
            raise AttributeError("Not data provided")
        tweets = pd.read_csv(data_file)

    prediction = model.predict(tweets)
    return prediction
