from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
import joblib
from statistics import mean
import logging
import sys
import getopt
from data import utils

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

training_filename = 'data/numerai_training_data.parquet'
validation_filename = 'data/numerai_validation_data.parquet'
#validation_predictions_filename = 'base_random_forest_validation_predictions.csv'
tournament_filename = 'data/numerai_tournament_data.parquet'
#tournament_predictions_filename = 'base_random_forest_tournament_predictions.csv'
#model_filename = 'base_random_forest.model'
validation_predictions_filename = 'predictions/neutralized_50_era_boosted_split_lgbm_regressor_validation_predictions.csv'
tournament_predictions_filename = 'predictions/neutralized_50_era_boosted_split_lgbm_regressor_tournament_predictions.csv'
model_filename_1 = 'models/era_boosted_split_lgbm_regressor_1.model'
model_filename_2 = 'models/era_boosted_split_lgbm_regressor_2.model'
model_filename_3 = 'models/era_boosted_split_lgbm_regressor_3.model'
model_filename_4 = 'models/era_boosted_split_lgbm_regressor_4.model'


target_col = 'target'


def read_data(filename):
    return pd.read_parquet(filename)


def corr(models, data, feature_names):
    prediction_column_names = [f'prediction_{i}' for i, model in enumerate(models)]
    for prediction_column_name, model in zip(prediction_column_names, models):
        data[prediction_column_name] = model.predict(data[feature_names])
    data['prediction'] = data[prediction_column_names].mean(axis=1)
    data['prediction'] = data['prediction'].rank(pct=True, method='first')
    correlation = np.corrcoef(data[target_col], data['prediction'])[0, 1]
    return correlation


def calc_corr(data):
    correlation = np.corrcoef(data['target'], data['prediction'])[0, 1]
    return correlation


def random_forest_model(training_data, feature_names, era_boosting_flag):
    logger.info("Training model")
    number_of_era_boosted_runs = 10
    params = {"n_estimators": 2000,
              "learning_rate": 0.01,
              "max_depth": 5,
              "num_leaves": 2 ** 5,
              "colsample_bytree": 0.1,
              "silent": False,
              "device_type": "gpu"}
    models = []
    model = LGBMRegressor(**params)
    model.fit(training_data[feature_names], training_data[target_col])
    models.append(model)
    if era_boosting_flag:
        eras = training_data.era.unique()
        while len(models) < number_of_era_boosted_runs:
            score_per_era = [(era, corr(models, training_data[training_data.era == era].copy(), feature_names)) for era in eras]
            ranked_eras = [era for era, score in sorted(score_per_era, key=lambda x: x[1])]
            era_training_data = training_data[training_data.era.isin(ranked_eras[:len(ranked_eras)//2])]
            model = LGBMRegressor(**params)
            model.fit(era_training_data[feature_names], era_training_data[target_col])
            models.append(model)
    return models


def train(era_boosting_flag):
    logger.info("Loading Training Data")
    training_data = read_data(training_filename)
    feature_names = [f for f in training_data.columns if 'feature' in f]
    training_data_1 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 0])].copy()
    training_data_2 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 1])].copy()
    training_data_3 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 2])].copy()
    training_data_4 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 3])].copy()
    del training_data
    models_1 = random_forest_model(training_data_1, feature_names, era_boosting_flag)
    logger.info("Saving models")
    for i, model in enumerate(models_1):
        joblib.dump(model, model_filename_1 + f'_{i}')
    del models_1
    models_2 = random_forest_model(training_data_2, feature_names, era_boosting_flag)
    logger.info("Saving models")
    for i, model in enumerate(models_2):
        joblib.dump(model, model_filename_2 + f'_{i}')
    del models_2
    models_3 = random_forest_model(training_data_3, feature_names, era_boosting_flag)
    logger.info("Saving models")
    for i, model in enumerate(models_3):
        joblib.dump(model, model_filename_3 + f'_{i}')
    del models_3
    models_4 = random_forest_model(training_data_4, feature_names, era_boosting_flag)
    logger.info("Saving models")
    for i, model in enumerate(models_4):
        joblib.dump(model, model_filename_4 + f'_{i}')
    logger.info("Done Training")


def validate():
    logger.info("Loading Validation Data")
    training_data = read_data(training_filename)
    feature_names = [f for f in training_data.columns if 'feature' in f]
    all_feature_corrs = training_data.groupby('era').apply(lambda d: d[feature_names].corrwith(d['target']))
    riskiest_features = utils.get_biggest_change_features(all_feature_corrs, 50)
    del(training_data)
    validation_data = read_data(validation_filename)
    print(validation_data)
    logger.info("Loading Models")
    models: list[LGBMRegressor] = []
    for i in range(10):
        models.append(joblib.load(model_filename_1 + f'_{i}'))
        models.append(joblib.load(model_filename_2 + f'_{i}'))
        models.append(joblib.load(model_filename_3 + f'_{i}'))
        models.append(joblib.load(model_filename_4 + f'_{i}'))
    logger.info("Predicting Validation Data")
    prediction_column_names = [f'prediction_{i}' for i, model in enumerate(models)]
    for prediction_column_name, model in zip(prediction_column_names, models):
        validation_data[prediction_column_name] = model.predict(validation_data[feature_names])
    validation_data['prediction_pre_neutralization'] = validation_data[prediction_column_names].mean(axis=1)
    validation_data['prediction_post_neutralization'] = utils.neutralize(df=validation_data,
                                                                         columns=['prediction_pre_neutralization'],
                                                                         neutralizers=riskiest_features,
                                                                         proportion=1.0,
                                                                         normalize=True,
                                                                         era_col='era')
    validation_data['prediction'] = validation_data['prediction_post_neutralization'].rank(pct=True, method='first')
    print(validation_data['prediction'])
    logger.info("Calculating Correlation")
    correlation = calc_corr(validation_data)
    logger.info(f"Correlation: {correlation}")
    logger.info("Saving Validation Data")
    validation_data['prediction'].to_csv(validation_predictions_filename, index=True)
    logger.info("Done Validating")


def predict():
    logger.info("Loading Training Data")
    training_data = read_data(training_filename)
    feature_names = [f for f in training_data.columns if 'feature' in f]
    logger.info("Determining Riskiest Features")
    all_feature_corrs = training_data.groupby('era').apply(lambda d: d[feature_names].corrwith(d['target']))
    riskiest_features = utils.get_biggest_change_features(all_feature_corrs, 50)
    del training_data
    logger.info("Loading Tournament Data")
    tournament_data = read_data(tournament_filename)
    logger.info("Loading Model")
    models: list[LGBMRegressor] = []
    for i in range(10):
        models.append(joblib.load(model_filename_1 + f'_{i}'))
        models.append(joblib.load(model_filename_2 + f'_{i}'))
        models.append(joblib.load(model_filename_3 + f'_{i}'))
        models.append(joblib.load(model_filename_4 + f'_{i}'))
    logger.info("Predicting Tournament Data")
    prediction_column_names = [f'prediction_{i}' for i, model in enumerate(models)]
    for prediction_column_name, model in zip(prediction_column_names, models):
        logger.info(f"Predicting {prediction_column_name}")
        tournament_data[prediction_column_name] = model.predict(tournament_data[feature_names])
    logger.info("Combining Predictions")
    tournament_data['prediction_pre_neutralization'] = tournament_data[prediction_column_names].mean(axis=1)
    logger.info("Neutralizing Predictions")
    tournament_data['prediction_post_neutralization'] = utils.neutralize(df=tournament_data,
                                                                         columns=['prediction_pre_neutralization'],
                                                                         neutralizers=riskiest_features,
                                                                         proportion=1.0,
                                                                         normalize=True,
                                                                         era_col='era')
    tournament_data['prediction'] = tournament_data['prediction_post_neutralization'].rank(pct=True, method='first')
    logger.info("Saving Tournament Data")
    tournament_data['prediction'].to_csv(tournament_predictions_filename, index=True)
    logger.info("Done Predicting")


def main(train_flag, validate_flag, predict_flag, era_boosting_flag):
    if train_flag:
        train(era_boosting_flag)

    if validate_flag:
        validate()

    if predict_flag:
        predict()


if __name__ == '__main__':
    argument_list = sys.argv[1:]
    short_options="tvpe"
    long_options=["train", "validate", "predict", "eraboost"]
    train_flag = False
    validate_flag = False
    predict_flag = False
    era_boosting_flag = False
    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
        for current_argument, current_value in arguments:
            if current_argument in ('-t', '--train'):
                print("Enabling Training")
                train_flag = True
            if current_argument in ('-e', '--eraboost'):
                print("Enabling Era Boosting during Training")
                era_boosting_flag = True
            elif current_argument in ('-v', '--validate'):
                print("Enabling Validation")
                validate_flag = True
            elif current_argument in ('-p', '--predict'):
                print("Enabling Predicting Tournament Data")
                predict_flag = True
        main(train_flag, validate_flag, predict_flag, era_boosting_flag)
    except getopt.error as e:
        print(str(e))
        sys.exit(2)
