import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict

def fit_round(rnd: int) -> Dict:
    """Send round number to client"""
    return {"rnd": rnd}

def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in
    # `evaluate` itself
    (X_test, y_test) = utils.load_credit_test_Default()
    (X_test_1, y_test_1) = utils.load_credit_test_Profit()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        #validation = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_VALIDATION_all1.csv")
        # col_n_k = ['PROFIT_LOSS']
        # validaiton_k = pd.DataFrame(validation, columns=col_n_k)
        # col_n_x = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL',
        #            'MNT_EPAR', 'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M',
        #            'NB_INTR_12M', 'PIR_DEL', 'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT',
        #            'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI', 'MNT_DEMANDE']
        # validaiton_x = pd.DataFrame(validation, columns=col_n_x)
        kk = y_test_1
        profit = 0

        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)

        predict = model.predict(X_test_1)
        for i in range(0, len(predict)):
            if predict[i - 1] == 1:
                kk[i:i + 1] = 0

        profit = sum(kk['PROFIT_LOSS'])
        print('accuracy: '+str(accuracy)+' profit: ' + str(profit))
        return loss, {"accuracy": 'accuracy: '+str(accuracy)+' profit: ' + str(profit)}

    return evaluate

model = LogisticRegression()
utils.set_initial_params(model)
strategy = fl.server.strategy.FedAvg(
    min_available_clients=4,
    eval_fn=get_eval_fn(model),
    on_fit_config_fn=fit_round,
)



fl.server.start_server(
    "0.0.0.0:8080",
    strategy=strategy,
    config={"num_rounds": utils.load_parameter_C()[2]}
)