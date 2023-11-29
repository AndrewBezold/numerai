from re import T
from typing import Optional, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from xgboost import Booster
from . data_generator import DataGenerator
import tensorflow as tf
import joblib
import logging
import sys
import gc
import scipy
import xgboost
import numerapi
import json
import os
from collections import defaultdict
from dotenv import load_dotenv


load_dotenv()


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


MODEL_DIR = 'models'
DATA_DIR = 'data'
PREDICTION_DIR = 'predictions'
TRAINING_PARQUET_FILENAME = 'train.parquet'
TRAINING_PARQUET_FILENAME_I = 'training_data_{i}.parquet'
VALIDATION_PARQUET_FILENAME = 'validation.parquet'
VALIDATION_PARQUET_FILENAME_I = 'validation_data_{i}.parquet'
TOURNAMENT_PARQUET_FILENAME = 'live.parquet'


'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
'''

INVALID_FEATURES = [
    'feature_palpebral_univalve_pennoncel',
    'feature_unsustaining_chewier_adnoun',
    'feature_brainish_nonabsorbent_assurance',
    'feature_coastal_edible_whang',
    'feature_disprovable_topmost_burrower',
    'feature_trisomic_hagiographic_fragrance',
    'feature_queenliest_childing_ritual',
    'feature_censorial_leachier_rickshaw',
    'feature_daylong_ecumenic_lucina',
    'feature_steric_coxcombic_relinquishment',
]


PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY = os.getenv('SECRET_KEY')


napi = numerapi.NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY, verbosity="warning")
current_round = None


def get_logger() -> logging.Logger:
    return logger


# DOWNLOAD DATA
def download_data(filename: str, version: str = 'v4.1', output_filename: Optional[str] = None) -> None:
    logger.info("Downloading Data")
    if output_filename is None:
        output_filename = filename
    napi.download_dataset(filename=f"{version}/{filename}", dest_path=f"{DATA_DIR}/{output_filename}")

def download_training_data(version: str = 'v4.1'):
    logger.info("Downloading Training Data")
    filename = 'train.parquet'
    download_data(filename, version)

def download_validation_data(version: str = 'v4.1'):
    logger.info("Downloading Validation Data")
    filename = 'validation.parquet'
    download_data(filename, version)

def shatter_validation_data():
    logger.info("Shattering Validation Data by Era")
    validation_data = load_validation_data()
    for era, group in validation_data.groupby('era'):
        save_parquet(group, path=f"{DATA_DIR}/validation_data_era_{era}.parquet")

def split_data(df: pd.DataFrame, base_filename: str, div: int = 4) -> None:
    for i in range(div):
        save_parquet(df[df['era'].astype(int) % div == i], base_filename.format(DATA_DIR=DATA_DIR, i=str(i)))

def download_tournament_data():
    logger.info("Downloading Tournament Data")
    filename = 'live.parquet'
    download_data(filename)

def download_training_example_predictions() -> str:
    logger.info("Downloading Training Example Predictions")
    filename = 'training_example_preds.parquet'
    download_data(filename, version='v4.1')
    return 'training_example_preds'

def download_validation_example_predictions() -> str:
    logger.info("Downloading Validation Example Predictions")
    filename = 'validation_example_preds.parquet'
    download_data(filename, version='v4.1')
    return 'validation_example_preds'

def download_live_example_predictions() -> str:
    logger.info("Downloading Live Example Predictions")
    filename = 'example_predictions.csv'
    download_data(filename, version='v3')
    return 'example_predictions'

def delete_data(filename: str) -> None:
    logger.info("Deleting Data")
    os.remove(f"{DATA_DIR}/{filename}")

def delete_tournament_data():
    logger.info("Deleting Tournament Data")
    filename = 'live.parquet'
    delete_data(filename)

def get_current_round() -> int:
    global current_round
    if current_round is None:
        logger.info("Getting Current Round")
        current_round = napi.get_current_round()
    return current_round

def get_account() -> dict:
    logger.info("Getting Account Data")
    account_info = napi.get_account()
    return account_info

def upload_diagnostics(name: str, model_id: str) -> None:
    logger.info("Uploading Diagnostics")
    filepath = f"{PREDICTION_DIR}/{name}.csv"
    napi.upload_diagnostics(file_path=filepath, model_id=model_id)

def upload_predictions(name: str, model_id: str) -> None:
    logger.info("Uploading Tournament Predictions")
    filepath = f"{PREDICTION_DIR}/{name}.csv"
    napi.upload_predictions(file_path=filepath, model_id=model_id)




# SAVE/LOAD STUFF
def save_pca(pca: PCA, name: str) -> None:
    logger.info("Saving PCA Model")
    joblib.dump(pca, f"{MODEL_DIR}/{name}.model")

def load_pca(name: str) -> PCA:
    logger.info("Loading PCA Model")
    pca = joblib.load(f"{MODEL_DIR}/{name}.model")
    return pca

def save_dict(d: Union[dict, list], name: str) -> None:
    logger.info("Saving Dict")
    filename = f"{MODEL_DIR}/{name}.json"
    with open(filename, 'w') as f:
        json.dump(d, f)

def load_dict(name: str) -> Union[dict, list]:
    logger.info("Loading Dict")
    filename = f"{MODEL_DIR}/{name}.json"
    with open(filename, 'r') as f:
        d = json.load(f)
    return d
    
def save_parquet(data: pd.DataFrame, path: str):
    logger.info("Saving Parquet")
    data.to_parquet(path)


# LOAD DATA
def read_parquet(filename: str) -> pd.DataFrame:
    return pd.read_parquet(filename)

def load_training_data(i=None) -> pd.DataFrame:
    logger.info("Loading Training Data")
    if i is None:
        filename = TRAINING_PARQUET_FILENAME
    else:
        filename = TRAINING_PARQUET_FILENAME_I.format(i=i)
    return read_parquet(f"{DATA_DIR}/{filename}")

def load_validation_data(i=None) -> pd.DataFrame:
    logger.info("Loading Validation Data")
    if i is None:
        filename = VALIDATION_PARQUET_FILENAME
    else:
        filename = VALIDATION_PARQUET_FILENAME_I.format(i=i)
    df = read_parquet(f"{DATA_DIR}/{filename}")
    df.drop(df[df['data_type'] == 'test'].index, inplace=True)
    return df

def load_tournament_data() -> pd.DataFrame:
    logger.info("Loading Tournament Data")
    df = read_parquet(f"{DATA_DIR}/{TOURNAMENT_PARQUET_FILENAME}")
    return df

def load_parquet(path) -> pd.DataFrame:
    logger.info("Loading Parquet")
    df = read_parquet(f"{DATA_DIR}/{path}")
    return df


def get_feature_names(data: pd.DataFrame) -> "list[str]":
    feature_names = [f for f in data.columns if 'feature' in f and f not in INVALID_FEATURES]
    return feature_names

def get_target(data: pd.DataFrame):
    return 'target_cyrus_v4_20'


# MODELS
## RANDOM FOREST
def create_random_forest_model() -> RandomForestRegressor:
    logger.info("Creating Random Forest Model")
    model = RandomForestRegressor(verbose=2, n_jobs=-1)
    return model

def train_random_forest_model(model: RandomForestRegressor, data: pd.DataFrame, feature_names: "list[str]", target: str) -> None:
    logger.info("Training Random Forest Model")
    model.fit(data[feature_names], data[target])

def predict_random_forest_model(model: RandomForestRegressor, data: pd.DataFrame, feature_names: "list[str]", prediction_column: str = 'prediction') -> None:
    logger.info("Predicting Random Forest Model")
    data[prediction_column] = model.predict(data[feature_names])

def save_random_forest_model(model: RandomForestRegressor, name: str) -> None:
    logger.info("Saving Random Forest Model")
    joblib.dump(model, f"{MODEL_DIR}/{name}.model")

def load_random_forest_model(name: str) -> RandomForestRegressor:
    logger.info("Loading Random Forest Model")
    return joblib.load(f"{MODEL_DIR}/{name}.model")

## LGBM
def create_lgbm_model(n_estimators=20000, max_depth=6) -> LGBMRegressor:
    logger.info("Creating LGBM Model")
    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.001,
        "max_depth": max_depth,
        "num_leaves": 2**max_depth,
        "colsample_bytree": 0.1,
        "verbosity": -1,
        "device_type": "gpu"
    }
    model = LGBMRegressor(**params)
    return model

def train_lgbm_model(model: LGBMRegressor, data: pd.DataFrame, feature_names: "list[str]", target: str) -> None:
    logger.info("Training LGBM Model")
    model.fit(data[feature_names], data[target])

def dothething(d: pd.DataFrame, model: LGBMRegressor, feature_names: "list[str]") -> pd.Series:
    r = model.predict(d[feature_names])
    #print("R")
    #print(r)
    s = pd.Series(r, index=d.index)
    #s = pd.Series(r)
    #print("S")
    #print(s)
    return s

def predict_lgbm_model(model: LGBMRegressor, data: pd.DataFrame, feature_names: "list[str]", prediction_column: str = 'prediction') -> None:
    logger.info("Predicting LGBM Model")
    #data[prediction_column] = data.groupby('era').apply(lambda d: pd.Series(model.predict(d[feature_names]), index=d.index))
    #test_output = data.groupby('era').apply(lambda d: dothething(d, model, feature_names))
    #print("Got past test output")
    #print(test_output)
    if len(data['era'].unique()) > 1:
        data[prediction_column] = data.groupby('era').apply(lambda d: dothething(d, model, feature_names)).reset_index(level='era', drop=True)
    else:
        data[prediction_column] = dothething(data, model, feature_names)

def save_lgbm_model(model: LGBMRegressor, name: str) -> None:
    logger.info("Saving LGBM Model")
    joblib.dump(model, f"{MODEL_DIR}/{name}.model")

def load_lgbm_model(name: str) -> LGBMRegressor:
    logger.info("Loading LGBM Model")
    return joblib.load(f"{MODEL_DIR}/{name}.model")

## XGBOOST
def create_xgboost_model(tree_method: str = 'gpu_hist') -> Booster:
    logger.info("Creating XGBoost Model")
    params = {
        "n_estimators": 20000,
        "learning_rate": 0.001,
        "max_depth": 6,
        "max_leaves": 2**6,
        "colsample_bytree": 0.1,
        "tree_method": tree_method,
        "verbosity": 2,
    }
    model = Booster(params=params)
    return model, params

def train_xgboost_model(model: Booster, params: dict, data: pd.DataFrame, feature_names: "list[str]", target: str, batch_size: int = 100) -> None:
    logger.info("Training XGBoost Model")
    gen = xgboost.DMatrix(DataGenerator(data, feature_names, target, batch_size), feature_names=feature_names)
    xgboost.train(params=params, num_boost_round=20000, dtrain=gen, xgb_model=model)

def predict_xgboost_model(model: Booster, data: pd.DataFrame, feature_names: "list[str]", prediction_column: str = 'prediction', target: str = 'target', batch_size: int = 100) -> None:
    logger.info("Predicting XGBoost Model")
    data[prediction_column] = model.predict(xgboost.DMatrix(data[feature_names]))

def save_xgboost_model(model: Booster, name: str) -> None:
    logger.info("Saving XGBoost Model")
    joblib.dump(model, f"{MODEL_DIR}/{name}.model")

def load_xgboost_model(name: str) -> Booster:
    logger.info("Loading XGBoost Model")
    return joblib.load(f"{MODEL_DIR}/{name}.model")


## Neural Net
def create_neural_net(input_size: int, output_size: int, hidden_sizes: "list[int]" = []) -> tf.keras.models.Sequential:
    logger.info("Creating Neural Net")
    hidden_layers = []
    for hidden_size in hidden_sizes:
        hidden_layers.append(tf.keras.layers.Dense(hidden_size, activation='tanh'))
        hidden_layers.append(tf.keras.layers.Dropout(0.2))
        #hidden_layers.append(tf.keras.layers.Dense(hidden_size, activation='selu', kernel_initializer='lecun_normal'))
        #hidden_layers.append(tf.keras.layers.AlphaDropout(0.2))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        *hidden_layers,
        tf.keras.layers.Dense(output_size, activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    #model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_neural_net(model: tf.keras.Model, data: pd.DataFrame, feature_names: "list[str]", targets: "list[str]", batch_size: int = 100, epochs: int = 10, callbacks = [], verbose: int = 1, dict_inputs: bool = False) -> None:
    logger.info("Training Neural Net")
    train_gen = DataGenerator(df=data, feature_names=feature_names, targets=targets, batch_size=batch_size, logger=logger, dict_inputs=True)
    model.fit(train_gen, epochs=epochs, verbose=verbose, callbacks=callbacks)

def predict_neural_net(model: tf.keras.Model, data: Union[pd.DataFrame, DataGenerator], feature_names: "list[str]", batch_size: Optional[int] = 100, prediction_column: str = 'prediction', inplace = True, verbose=1, multiple_outputs=True, dict_inputs: bool = False) -> Optional[pd.Series]:
    if verbose > 0:
        logger.info("Predicting Neural Net")
    if batch_size == None:
        batch_size = len(data.index)
    if inplace:
        data[prediction_column] = 0.5
    if type(data) == DataGenerator:
        test_gen = data
    else:
        test_gen = DataGenerator(df=data, feature_names=feature_names, batch_size=batch_size, shuffle=False, dict_inputs=True)
    if inplace:
        if multiple_outputs:
            data[prediction_column] = model.predict(test_gen, verbose=verbose)[-1]
        else:
            data[prediction_column] = model.predict(test_gen, verbose=verbose)
        #for i in range(len(test_gen)):
        #    data.loc[data.index[i*batch_size:(i+1)*batch_size], prediction_column] = model.predict(test_gen[i][0])
    else:
        results_list = []
        for i in range(len(test_gen)):
            results_list.append(model.predict(test_gen[i][0]))
            gc.collect()
        results = np.concatenate(results_list, axis=None)
        return pd.Series(results)

def save_neural_net(model: tf.keras.models.Sequential, name: str) -> None:
    logger.info("Saving Neural Net")
    model.save(f"{MODEL_DIR}/{name}.tf")

def load_neural_net(name: str) -> tf.keras.models.Sequential:
    logger.info("Loading Neural Net")
    return tf.keras.models.load_model(f"{MODEL_DIR}/{name}.tf")


# STANDARDIZE PREDICTIONS
def feature_neutralization():
    pass

def rank(data: pd.DataFrame, prediction_column: str = 'prediction') -> pd.DataFrame:
    logger.info("Ranking Predictions")
    if len(data['era'].unique()) > 1:
        data[prediction_column] = data.groupby('era').apply(lambda d: d[prediction_column].rank(pct=True, method='first')).reset_index(level='era', drop=True)
    else:
        data[prediction_column] = data[prediction_column].rank(pct=True, method='first')
    return data


# SAVE PREDICTIONS
def save_predictions(data: pd.DataFrame, name: str, prediction_column: str = 'prediction'):
    logger.info("Saving Predictions")
    if prediction_column != 'prediction':
        data['prediction'] = data[prediction_column]
    data['prediction'].to_csv(f"{PREDICTION_DIR}/{name}.csv", index=True)
    if prediction_column != 'prediction':
        del data['prediction']

def load_predictions(name: str, prediction_column: str = 'prediction') -> pd.DataFrame:
    logger.info("Loading Predictions")
    df = pd.read_csv(f"{PREDICTION_DIR}/{name}.csv", index_col='id')
    if prediction_column != 'prediction':
        df.rename(columns={'prediction': prediction_column}, inplace=True)
    print(df.head())
    return df

def load_example_predictions(name: str) -> pd.DataFrame:
    logger.info("Loading Example Predictions")
    df = pd.read_parquet(f"{DATA_DIR}/{name}.parquet")
    df.rename(columns={'prediction': 'example_predictions'}, inplace=True)
    return df


# STATISTICS
def corr(data: pd.DataFrame, model, feature_names, target: str = 'target', riskiest_features = []):
    predict_neural_net(model, data, feature_names, batch_size=10000, verbose=0)
    #data['prediction'] = neutralize(data, ['prediction'], riskiest_features, proportion=1.0, normalize=True)
    data['prediction'] = data['prediction'].rank(pct=True, method='first')
    corr_predictions(data, 'prediction', target)
    del data['prediction']

def corr_predictions(data: pd.DataFrame, prediction_column: str = 'prediction', target: str = 'target'):
    validation_correlations = data.groupby('era').apply(lambda d: d[prediction_column].corr(d[target]))
    mean = validation_correlations.mean()
    std = validation_correlations.std(ddof=0)
    sharpe = mean / std
    print(f"Corr: {mean}")
    print(f"Sharpe: {sharpe}")

def neutralize(df, columns, neutralizers=None, proportion=1.0, normalize=True):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df['era'].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df['era'] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


# NEW WEIRD STUFF

# get coefficients of features in linear model, plug into tsne, add as new features
def linear_coef(data: pd.DataFrame, feature_names, target, pca=None):
    logger.info("Getting Era-Wise Coefficients")
    rlm = Ridge(fit_intercept=False)
    unique_eras = data['era'].unique()
    #k = 150
    k = 30
    coefs = data.groupby('era').apply(lambda d: coef_features(d[feature_names], d[target], rlm))
    pca_features, pca = get_pca(coefs, k, pca)
    pca_feature_names = [f"pca_{i+1}" for i in range(len(pca_features[0]))]
    print(len(pca_feature_names))
    data[pca_feature_names] = data.groupby('era').apply(temp_lambda, pca_features, unique_eras, pca_feature_names)
    print(data[['era'] + pca_feature_names])
    return pca_feature_names, pca


def temp_lambda(d, pca_features, unique_eras, pca_feature_names):
    pca_features_this_era = pca_features[np.where(unique_eras == d.name)][0]
    return pd.DataFrame(np.tile(pca_features_this_era, (len(d.index), 1)), columns=pca_feature_names, index=d.index)

def coef_features(features, target, rlm: Ridge):
    features = features.values - .5
    target = target - .5
    rlm.fit(features, target)
    coefs = rlm.coef_
    return pd.Series(coefs)

def get_pca(coefs, k, pca: PCA):
    if pca is None:
        pca = PCA(n_components=k)
        pca.fit(coefs)
    pca_features = pca.transform(coefs)
    return pca_features, pca


def generate_feature_correlations(data: pd.DataFrame, feature_names: "list[str]") -> "dict[str, dict[str, float]]":
    logger.info("Generating Feature Correlations")
    logger.info(f"Total Number of Features: {len(feature_names)}")
    correlations = defaultdict(dict)
    for i, feature_1 in enumerate(feature_names):
        logger.info(f"Feature 1: {i}")
        for j, feature_2 in enumerate(feature_names[i+1:]):
            #logger.info(f"Feature 1: {i}; Feature 2: {j}")
            corr = data[feature_1].corr(data[feature_2])
            correlations[feature_1][feature_2] = corr
            correlations[feature_2][feature_1] = corr
    logger.info("Done Generating Feature Correlations")
    return correlations


def save_feature_correlations(feature_correlations: "dict[str, dict[str, float]]"):
    logger.info("Saving Feature Correlations")
    filename = f"{MODEL_DIR}/utils/feature_correlations.json"
    with open(filename, 'w') as f:
        json.dump(feature_correlations, f)

def load_feature_correlations() -> "dict[str, dict[str, float]]":
    logger.info("Loading Feature Correlations")
    filename = f"{MODEL_DIR}/utils/feature_correlations.json"
    with open(filename, 'r') as f:
        feature_correlations = json.load(f)
    return feature_correlations
