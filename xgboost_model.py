from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
import joblib
from statistics import mean
import logging
import sys
import getopt

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
validation_predictions_filename = 'era_boosted_split_10_random_forest_validation_predictions.csv'
tournament_predictions_filename = 'era_boosted_split_random_forest_tournament_predictions_csv'
model_filename_1 = 'era_boosted_split_10_random_forest_1.model'
model_filename_2 = 'era_boosted_split_10_random_forest_2.model'
model_filename_3 = 'era_boosted_split_10_random_forest_3.model'
model_filename_4 = 'era_boosted_split_10_random_forest_4.model'


def read_data(filename):
    return pd.read_parquet(filename)


def corr(model, data, feature_names):
    predictions = model.predict(data[feature_names])
    data['prediction'] = predictions
    data['prediction'] = data['prediction'].rank(pct=True, method='first')
    correlation = np.corrcoef(data['target'], data['prediction'])[0, 1]
    return correlation


def random_forest_model(training_data, feature_names, era_boosting_flag):
    logger.info("Training model")
    era_boosting_runs = 10
    trees = 10 if era_boosting_flag else 100
    model = RandomForestRegressor(verbose=2, n_jobs=-1, max_depth=6, n_estimators=trees, warm_start=era_boosting_flag)
    model.fit(training_data[feature_names], training_data['target'])
    if era_boosting_flag:
        eras = training_data.era.unique()
        while model.n_estimators < 100:
            score_per_era = [(era, corr(model, training_data[training_data.era == era].copy(), feature_names)) for era in eras]
            ranked_eras = [era for era, score in sorted(score_per_era, key=lambda x: x[1])]
            era_training_data = training_data[training_data.era.isin(ranked_eras[:int(len(ranked_eras)/10)])]
            model.n_estimators += 10
            model.fit(era_training_data[feature_names], era_training_data['target'])
    return model


def train(era_boosting_flag):
    logger.info("Loading Training Data")
    training_data = read_data(training_filename)
    feature_names = [f for f in training_data.columns if 'feature' in f]
    training_data_1 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 0])].copy()
    training_data_2 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 1])].copy()
    training_data_3 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 2])].copy()
    training_data_4 = training_data[training_data.era.isin([era for era in training_data.era.unique() if int(era) % 4 == 3])].copy()
    del training_data
    model_1 = random_forest_model(training_data_1, feature_names, era_boosting_flag)
    logger.info("Saving model")
    joblib.dump(model_1, model_filename_1)
    del model_1
    model_2 = random_forest_model(training_data_2, feature_names, era_boosting_flag)
    logger.info("Saving model")
    joblib.dump(model_2, model_filename_2)
    del model_2
    model_3 = random_forest_model(training_data_3, feature_names, era_boosting_flag)
    logger.info("Saving model")
    joblib.dump(model_3, model_filename_3)
    del model_3
    model_4 = random_forest_model(training_data_4, feature_names, era_boosting_flag)
    logger.info("Saving model")
    joblib.dump(model_4, model_filename_4)
    logger.info("Done Training")


def validate():
    logger.info("Loading Validation Data")
    validation_data = read_data(validation_filename)
    feature_names = [f for f in validation_data.columns if 'feature' in f]
    print(validation_data)
    logger.info("Loading Models")
    model_1 = joblib.load(model_filename_1)
    model_2 = joblib.load(model_filename_2)
    model_3 = joblib.load(model_filename_3)
    model_4 = joblib.load(model_filename_4)
    logger.info("Predicting Validation Data")
    validation_data['prediction_1'] = model_1.predict(validation_data[feature_names])
    validation_data['prediction_2'] = model_2.predict(validation_data[feature_names])
    validation_data['prediction_3'] = model_3.predict(validation_data[feature_names])
    validation_data['prediction_4'] = model_4.predict(validation_data[feature_names])
    validation_data['prediction'] = validation_data[['prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].mean(axis=1)
    validation_data['prediction'] = validation_data['prediction'].rank(pct=True, method='first')
    print(validation_data['prediction'])
    logger.info("Saving Validation Data")
    validation_data['prediction'].to_csv(validation_predictions_filename, index=True)
    logger.info("Done Validating")


def predict():
    logger.info("Loading Tournament Data")
    tournament_data = read_data(tournament_filename)
    feature_names = [f for f in tournament_data.columns if 'feature' in f]
    logger.info("Loading Model")
    model = joblib.load(model_filename_1)
    logger.info("Predicting Tournament Data")
    predictions = model.predict(tournament_data[feature_names])
    tournament_data['prediction'] = predictions
    tournament_data['prediction'] = tournament_data['prediction'].rank(pct=True, method='first')
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
