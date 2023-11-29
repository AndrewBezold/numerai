import utils
import pandas as pd
import numpy as np
import tensorflow as tf
from tf_callbacks import Callback
from methods import (
    random_forest,
    xgboost,
    lgbm,
    neural_net,
)

#gpu = tf.config.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)

def gpu_xgboost():
    data = utils.load_training_data()
    feature_names = utils.get_feature_names(data)
    target = utils.get_target(data)
    model = utils.create_xgboost_model(tree_method='gpu_hist')
    utils.train_xgboost_model(model, data, feature_names, target)
    utils.save_xgboost_model(model, "gpu_xgboost")
    del data
    data = utils.load_validation_data()
    prediction_column = 'xgboost_prediction'
    utils.predict_xgboost_model(model, data, feature_names, prediction_column)
    data = utils.rank(data, prediction_column)
    utils.save_predictions(data, "gpu_xgboost_validation", prediction_column)
