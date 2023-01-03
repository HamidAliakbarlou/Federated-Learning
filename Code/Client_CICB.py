import warnings
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

# Load dataset
(X_train_3, y_train_3) = utils.load_credit_train_3_Default()
(X_test, y_test) = utils.load_credit_test_Default()

# Split train set into 10 partitions and randomly use one for training.
#partition_id = np.random.choice(10)
#(X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

# Create LogisticRegression Model
model = LogisticRegression(
    penalty="l2",
    max_iter=utils.load_parameter_C()[0], # local epoch
    warm_start=True, # prevent refreshing weights when fitting
    C=utils.load_parameter_C()[1]
)

# Setting initial parameters, akin to model.compile for keras models
utils.set_initial_params(model)

class MnistClient(fl.client.NumPyClient):
    def get_parameters(self): # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_3, y_train_3)
            print(f"Training finished for round {config['rnd']}")
        return utils.get_model_parameters(model), len(X_train_3), {}

    def evaluate(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

fl.client.start_numpy_client("0.0.0.0:8080", client=MnistClient())