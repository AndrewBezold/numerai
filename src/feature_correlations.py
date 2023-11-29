from . import utils


def main():
    training_data = utils.load_training_data()
    feature_names = utils.get_feature_names(training_data)
    feature_correlations = utils.generate_feature_correlations(training_data, feature_names)
    utils.save_feature_correlations(feature_correlations)


if __name__ == '__main__':
    main()