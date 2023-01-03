from typing import Tuple, Union, List
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]
#Load Data
test = pd.read_csv("credit/CreditGame_VALIDATION_all.csv")
#test = pd.read_csv("credit/CreditGame_TEST_all.csv")
train_1 = pd.read_csv("credit/CreditGame_TRAIN_1.csv")
train_2 = pd.read_csv("credit/CreditGame_TRAIN_2.csv")
train_3 = pd.read_csv("credit/CreditGame_TRAIN_3.csv")
train_4 = pd.read_csv("credit/CreditGame_TRAIN_4.csv")

def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params

def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 2 # MNIST has 10 classes
    n_features = 27 # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def load_credit_test_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y = ['DEFAULT']
    y_test = pd.DataFrame(test, columns=col_n_y)

    col_n_x = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_test = pd.DataFrame(test, columns=col_n_x)

    return (x_test, y_test)

def load_credit_test_Profit() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_k = ['PROFIT_LOSS']
    y_test_1 = pd.DataFrame(test, columns=col_n_y_k)

    col_n_x_k = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_test_1 = pd.DataFrame(test, columns=col_n_x_k)

    return (x_test_1, y_test_1)

def load_credit_train_1_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_1 = ['DEFAULT']
    y_train_1 = pd.DataFrame(train_1, columns=col_n_y_1)

    col_n_x_1 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_train_1 = pd.DataFrame(train_1, columns=col_n_x_1)

    return (x_train_1, y_train_1)

def load_credit_train_2_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_2 = ['DEFAULT']
    y_train_2 = pd.DataFrame(train_2, columns=col_n_y_2)

    col_n_x_2 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_train_2 = pd.DataFrame(train_2, columns=col_n_x_2)

    return (x_train_2, y_train_2)

def load_credit_train_3_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_3 = ['DEFAULT']
    y_train_3 = pd.DataFrame(train_3, columns=col_n_y_3)

    col_n_x_3 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_train_3 = pd.DataFrame(train_3, columns=col_n_x_3)

    return (x_train_3, y_train_3)

def load_credit_train_4_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_4 = ['DEFAULT']
    y_train_4 = pd.DataFrame(train_4, columns=col_n_y_4)

    col_n_x_4 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_train_4 = pd.DataFrame(train_4, columns=col_n_x_4)

    return (x_train_4, y_train_4)

def load_parameter_C():
    """
    Loads the parameter
    """
    #client iteration
    max_iter = 1
    #logistic regression regulization parameter
    C = 1
    #server round
    round = 20

    return (max_iter,C,round)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
        np.array_split(y, num_partitions))
    )

