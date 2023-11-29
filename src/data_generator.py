from typing import Callable, Optional, Union
import tensorflow as tf
import pandas as pd
import numpy as np
import xgboost
import os
import logging

class DataGenerator(tf.keras.utils.Sequence, xgboost.DataIter):
    def __init__(self, df: pd.DataFrame, feature_names: "list[str]", targets: "Union[str, list[str]]" = [], batch_size: int = 100, shuffle: bool = True, logger: Optional[logging.Logger] = None, dict_inputs: bool = False):
        self.df = df
        self.feature_names = feature_names
        if type(targets) == str:
            self.targets = [targets]
        else:
            self.targets = targets
        self.batch_size = batch_size
        self._it = 0
        self.shuffle = shuffle
        self.logger = logger
        self.dict_inputs = dict_inputs
        super(DataGenerator, self).__init__()
        self.reset()

    def __len__(self):
        return int(np.ceil(len(self.df.index) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx+1) * self.batch_size]
        if self.dict_inputs:
            inputs = {feature_name: batch[[feature_name]] for feature_name in self.feature_names}
        else:
            inputs = batch[self.feature_names]
        if self.targets:
            return (inputs, batch[self.targets])
        return (inputs,)

    def next(self, input_data: Callable):
        if self._it == len(self):
            return 0

        batch = self[self._it]
        input_data(data=batch[0], label=batch[1])
        self._it += 1
        return 1

    def reset(self):
        self._it = 0
        if self.shuffle:
            if self.logger is not None:
                self.logger.info("Shuffling Data")
            self.df = self.df.sample(frac=1)

    def on_epoch_end(self):
        self.reset()
