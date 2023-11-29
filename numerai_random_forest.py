from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
import joblib
from statistics import mean
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

training_filename = 'data/numerai_training_data.csv'
testing_filename = 'data/numerai_tournament_data.csv'
predictions_filename = 'base_random_forest_predictions.csv'
model_filename = 'base_random_forest.model'

classes = [0.0, 0.25, 0.5, 0.75, 1.0]


def read_csv(filename):
    return pd.read_csv(filename, index_col=0)


def convert(raw_data, feature_names, training=False):
    if training:
        raw_data[feature_names] = raw_data[feature_names].apply(pd.to_numeric, downcast='float')
        raw_data = pd.concat([raw_data[raw_data.target == 0.00]] * 20 + [raw_data[raw_data.target == 0.25]] * 5 + [raw_data[raw_data.target == 0.50]] * 2 + [raw_data[raw_data.target == 0.75]] * 5 + [raw_data[raw_data.target == 1.00]] * 20)
        logger.info(f"Raw Data: {raw_data}")
    data_features = raw_data[feature_names]
    data_targets_raw = raw_data['target']
    data_targets = pd.get_dummies(data_targets_raw)
    logger.info(f"Targets: {data_targets}")
    column_names = data_targets.columns.values
    logger.info(f"Column names: {column_names}")
    '''
    results = {}
    for target in data_targets:
        if target not in results:
            results[target] = 0
        results[target] += 1
    logger.info(f"Targets: {results}")
    '''
    return data_features, data_targets


def score_model(model, validation_features, validation_target):
    logger.info("Validating model")
    predictions = np.asarray(model.predict_proba(validation_features))[:,:,1].transpose()
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Targets: {validation_target}")
    '''
    results = {}
    for prediction in predictions:
        if prediction not in results:
            results[prediction] = 0
        results[prediction] += 1
    logger.info(f"Results: {results}")
    '''
    one_hot = [[1 if i == max(prediction) else 0 for i in prediction] for prediction in predictions]
    one_hot_accuracy = mean([1 if validation_target.iloc[i].values.tolist() == prediction else 0 for i, prediction in enumerate(one_hot)])
    logger.info(f"One Hot Accuracy: {one_hot_accuracy}")
    accuracy = 1.0 - mean_absolute_error(validation_target, predictions)
    logger.info(f"Accuracy: {accuracy}")
    logloss = log_loss(validation_target, predictions)
    logger.info(f"Logloss: {logloss}")


def score_separate_models(model_000, model_025, model_050, model_075, model_100, validation_data, validation_target):
    logger.info("Validating model")
    predictions_000 = model_000.predict_proba(validation_data)[:,1]
    predictions_025 = model_025.predict_proba(validation_data)[:,1]
    predictions_050 = model_050.predict_proba(validation_data)[:,1]
    predictions_075 = model_075.predict_proba(validation_data)[:,1]
    predictions_100 = model_100.predict_proba(validation_data)[:,1]
    logger.info(f"One prediction set: {predictions_000}")
    predictions = np.asarray([np.asarray([predictions_000[i], predictions_025[i], predictions_050[i], predictions_075[i], predictions_100[i]])/sum([predictions_000[i], predictions_025[i], predictions_050[i], predictions_075[i], predictions_100[i]]) for i, _ in enumerate(predictions_000)])
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Targets: {validation_target}")
    one_hot = [[1 if i == max(prediction) else 0 for i in prediction] for prediction in predictions]
    one_hot_accuracy = mean([1 if validation_target.iloc[i].values.tolist() == prediction else 0 for i, prediction in enumerate(one_hot)])
    logger.info(f"One Hot Accuracy: {one_hot_accuracy}")
    accuracy = 1.0 - mean_absolute_error(validation_target, predictions)
    logger.info(f"Accuracy: {accuracy}")
    logloss = log_loss(validation_target, predictions)
    logger.info(f"Logloss: {logloss}")


def score_forest_waves_model(primary_model, inner_model, outer_model, validation_data, validation_target):
    logger.info("Validating model")
    predictions_primary = np.asarray(primary_model.predict_proba(validation_data))[:,:,1].transpose()
    predictions_inner = np.asarray(inner_model.predict_proba(validation_data))[:,:,1].transpose()
    predictions_outer = np.asarray(outer_model.predict_proba(validation_data))[:,:,1].transpose()
    logger.info(f"Primary Predictions: {predictions_primary}\nInner Predictions: {predictions_inner}\nOuter Predictions: {predictions_outer}")
    predictions = np.asarray([np.asarray([(predictions_primary[i][0] + predictions_primary[i][4]) * predictions_outer[i][0], (predictions_primary[i][1] + predictions_primary[i][3]) * predictions_inner[i][0], predictions_primary[i][2], (predictions_primary[i][3] + predictions_primary[i][1]) * predictions_inner[i][1], (predictions_primary[i][4] + predictions_primary[i][0]) * predictions_outer[i][1]]) for i, _ in enumerate(predictions_primary)])
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Targets: {validation_target}")
    one_hot = [[1 if i == max(prediction) else 0 for i in prediction] for prediction in predictions]
    one_hot_accuracy = mean([1 if validation_target.iloc[i].values.tolist() == prediction else 0 for i, prediction in enumerate(one_hot)])
    logger.info(f"One Hot Accuracy: {one_hot_accuracy}")
    accuracy = 1.0 - mean_absolute_error(validation_target, predictions)
    logger.info(f"Accuracy: {accuracy}")
    logloss = log_loss(validation_target, predictions)
    logger.info(f"Logloss: {logloss}")


def weighted_result(a):
    return a[0] * 0 + a[1] * 0.25 + a[2] * 0.5 + a[3] * 0.75 + a[4] * 1


def predict_tournament(model, tournament_data):
    logger.info("Predicting tournament data")
    predictions = np.asarray(model.predict_proba(tournament_data))[:,:,1].transpose()
    logger.info("Saving predictions")
    logger.info(predictions)
    tournament_data['prediction'] = np.apply_along_axis(weighted_result, 1, predictions)
    logger.info(tournament_data['prediction'])
    tournament_data['prediction'].to_csv(predictions_filename)
    logger.info("Done")

def predict_separate_models(model_000, model_025, model_050, model_075, model_100, tournament_data):
    logger.info("Predicting tournament data")
    predictions_000 = model_000.predict_proba(tournament_data)[:,1]
    predictions_025 = model_025.predict_proba(tournament_data)[:,1]
    predictions_050 = model_050.predict_proba(tournament_data)[:,1]
    predictions_075 = model_075.predict_proba(tournament_data)[:,1]
    predictions_100 = model_100.predict_proba(tournament_data)[:,1]
    predictions = np.asarray([np.asarray([predictions_000[i], predictions_025[i], predictions_050[i], predictions_075[i], predictions_100[i]])/sum([predictions_000[i], predictions_025[i], predictions_050[i], predictions_075[i], predictions_100[i]]) for i, _ in enumerate(predictions_000)])
    logger.info("Saving predictions")
    logger.info(predictions)
    tournament_data['prediction'] = np.apply_along_axis(weighted_result, 1, predictions)
    logger.info(tournament_data['prediction'])
    tournament_data['prediction'].to_csv('separate_forests_predictions.csv')


def predict_forest_waves_model(primary_model, inner_model, outer_model, tournament_data):
    logger.info("Predicting tournament data")
    predictions_primary = np.asarray(primary_model.predict_proba(tournament_data))[:,:,1].transpose()
    predictions_inner = np.asarray(inner_model.predict_proba(tournament_data))[:,:,1].transpose()
    predictions_outer = np.asarray(outer_model.predict_proba(tournament_data))[:,:,1].transpose()
    predictions = np.asarray([np.asarray([(predictions_primary[i][0] + predictions_primary[i][4]) * predictions_outer[i][0], (predictions_primary[i][1] + predictions_primary[i][3]) * predictions_inner[i][0], predictions_primary[i][2], (predictions_primary[i][3] + predictions_primary[i][1]) * predictions_inner[i][1], (predictions_primary[i][4] + predictions_primary[i][0]) * predictions_outer[i][1]]) for i, _ in enumerate(predictions_primary)])
    logger.info("Saving predictions")
    logger.info(predictions)
    tournament_data['prediction'] = np.apply_along_axis(weighted_result, 1, predictions)
    logger.info(tournament_data['prediction'])
    tournament_data['prediction'].to_csv('forest_waves_predictions.csv')


def random_forest_model(training_data, training_targets):
    logger.info("Training model")
    model = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    model.fit(training_data, training_targets)
    return model


def separate_random_forest_model_per_target(training_data, training_targets):
    logger.info("Training models")

    logger.info("Training Model 000")
    model_000 = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    model_000.fit(training_data, training_targets[0.00])

    logger.info("Training Model 025")
    model_025 = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    model_025.fit(training_data, training_targets[0.25])

    logger.info("Training Model 050")
    model_050 = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    model_050.fit(training_data, training_targets[0.50])

    logger.info("Training Model 075")
    model_075 = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    model_075.fit(training_data, training_targets[0.75])

    logger.info("Training Model 100")
    model_100 = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    model_100.fit(training_data, training_targets[1.00])

    return model_000, model_025, model_050, model_075, model_100


def forest_waves_model(training_data, training_targets):
    logger.info("Training models")
    training_targets_inner = training_targets[(training_targets[0.25] == 1) | (training_targets[0.75] == 1)][[0.25, 0.75]]
    training_data_inner = training_data[training_data.index.isin(training_targets_inner.index.array)]
    logger.info(f"Training Data Inner: {training_targets_inner}\n{training_data_inner}")

    training_targets_outer = training_targets[(training_targets[0.00] == 1) | (training_targets[1.00] == 1)][[0.00, 1.00]]
    training_data_outer = training_data[training_data.index.isin(training_targets_outer.index.array)]
    logger.info(f"Training Data Outer: {training_targets_outer}\n{training_data_outer}")

    logger.info("Training Primary Model")
    primary_model = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    primary_model.fit(training_data, training_targets)

    logger.info("Training Inner Model")
    inner_model = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    inner_model.fit(training_data_inner, training_targets_inner)

    logger.info("Training Outer Model")
    outer_model = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6)
    outer_model.fit(training_data_outer, training_targets_outer)
    
    return primary_model, inner_model, outer_model


def corr(predictions: pd.DataFrame, targets):
    ranked_predictions = predictions.rank(pct=True, method="first")
    correlation = np.corrcoef(targets, ranked_predictions)[0, 1]
    return correlation


def era_boosting(training_data: pd.DataFrame, training_targets, feature_names):
    training_data_by_era = [y for x, y in training_data.groupby('era', as_index=False)]
    model = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth=6, n_estimators=10)
    model.fit(training_data[feature_names], training_targets)
    predictions_by_era = [model.predict_proba(era_training_data)[:,:,1].transpose() for era_training_data in training_data_by_era]
    correlations = [corr(prediction, training_data_by_era.iloc[i]['target']) for i, prediction in enumerate(predictions_by_era)]
    pass


def main():
    # Load a dataset in a Pandas dataframe.
    logger.info("Loading data")
    training_data_raw = read_csv(training_filename)
    tournament_data_raw = read_csv(testing_filename)
    feature_names = [f for f in training_data_raw.columns if 'feature' in f]
    training_data, training_targets = convert(training_data_raw, feature_names, training=True)
    del training_data_raw
    validation_data_raw = tournament_data_raw[tournament_data_raw.data_type == "validation"]
    validation_data, validation_targets = convert(validation_data_raw, feature_names)
    tournament_data, _ = convert(tournament_data_raw, feature_names)
    del tournament_data_raw

    # Train a Random Forest model.
    #model = random_forest_model(training_data, training_targets)
    #model_000, model_025, model_050, model_075, model_100 = separate_random_forest_model_per_target(training_data, training_targets)
    primary_model, inner_model, outer_model = forest_waves_model(training_data, training_targets)

    # Validate
    #score_model(model, validation_data, validation_targets)
    #score_separate_models(model_000, model_025, model_050, model_075, model_100, validation_data, validation_targets)
    score_forest_waves_model(primary_model, inner_model, outer_model, validation_data, validation_targets)

    # Export the model to a SavedModel.
    logger.info("Saving model")
    #joblib.dump(model, model_filename)
    #joblib.dump(model_000, 'split_forests_000.model')
    #joblib.dump(model_025, 'split_forests_025.model')
    #joblib.dump(model_050, 'split_forests_050.model')
    #joblib.dump(model_075, 'split_forests_075.model')
    #joblib.dump(model_100, 'split_forests_100.model')
    joblib.dump(primary_model, 'forest_waves_primary.model')
    joblib.dump(inner_model, 'forest_waves_inner.model')
    joblib.dump(outer_model, 'forest_waves_outer.model')

    #predict_tournament(model, tournament_data)
    #predict_separate_models(model_000, model_025, model_050, model_075, model_100, tournament_data)
    predict_forest_waves_model(primary_model, inner_model, outer_model, tournament_data)


if __name__ == '__main__':
    main()
