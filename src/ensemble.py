from . methods import neural_net, lgbm
from . import utils
import pandas as pd

def ensemble():
    training_data = utils.load_training_data()
    feature_names = utils.get_feature_names(training_data)
    target = utils.get_target(training_data)
    #neural_net.train_baseline_neural_net(training_data=training_data, validation_data=None, feature_names=feature_names, target=target, callback=False)
    #lgbm.train_baseline_lgbm(data=training_data, feature_names=feature_names, target=target)
    del training_data
    validation_data = utils.load_validation_data()
    neural_net.validate_baseline_neural_net(data=validation_data, feature_names=feature_names, prediction_column='neural_net_prediction')
    lgbm.validate_baseline_lgbm(data=validation_data, feature_names=feature_names, prediction_column='lgbm_prediction')
    # compare models correlation with each other
    print("Correlating predictions")
    print("Neural Net - Target")
    utils.corr_predictions(validation_data, 'neural_net_prediction', 'target')
    print("LGBM - Target")
    utils.corr_predictions(validation_data, 'lgbm_prediction', 'target')
    print("Neural Net - LGBM")
    utils.corr_predictions(validation_data, 'neural_net_prediction', 'lgbm_prediction')
    # ensemble validation predictions
    print("Ensembling predictions")
    validation_data['prediction'] = validation_data[['neural_net_prediction', 'lgbm_prediction']].mean(axis=1)
    utils.rank(validation_data, 'prediction')
    utils.save_predictions(validation_data, 'validation_ensemble')

if __name__ == '__main__':
    ensemble()