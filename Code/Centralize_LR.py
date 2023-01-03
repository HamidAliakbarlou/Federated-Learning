import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy import *

#Centralize Logistic Regression: RBC, BMO, TD, CICB, Soctia

#0.Load Data
#train & test data
train_all = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_all.csv")
test_all = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TEST_all.csv")
#column name
col_n_y = ['DEFAULT']
col_n_x = ['NB_EMPT', 'R_ATD','DUREE','PRT_VAL','AGE_D', 'REV_BT', 'REV_NET','TYP_RES', 'ST_EMPL', 'MNT_EPAR', 'NB_ER_6MS', 'NB_ER_12MS','NB_DEC_12MS',  'NB_OPER','NB_COUR','NB_INTR_1M','NB_INTR_12M','PIR_DEL','NB_DEL_30','NB_DEL_60', 'NB_DEL_90','MNT_PASS','MNT_ACT', 'MNT_AUT_REN','MNT_UTIL_REN', 'NB_SATI', 'MNT_DEMANDE']
col_n_k = ['PROFIT_LOSS']

#1.RBC Full data
#get full train data
train_y = pd.DataFrame(train_all, columns=col_n_y)
train_x = pd.DataFrame(train_all, columns=col_n_x)
test_y = pd.DataFrame(test_all, columns=col_n_y)
test_x = pd.DataFrame(test_all, columns=col_n_x)
test_k = pd.DataFrame(test_all, columns=col_n_k)
#modle
#Default
clf_all = LogisticRegression(random_state=0).fit(train_x, ravel(train_y))
predict_all = clf_all.predict(test_x)
print(np.mean(predict_all==ravel(test_y)))
#Profit
kk = test_k
for i in range(0,len(predict_all)):
      if predict_all[i-1]==1:
          kk[i:i+1] = 0
print(sum(kk['PROFIT_LOSS']))

#0.6932666666666667
#107960

#Client
#client 1
train_1 = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_1.csv")
train_x_1 = pd.DataFrame(train_1, columns=col_n_x)
train_y_1 = pd.DataFrame(train_1, columns=col_n_y)
#model 1
#Default
clf_all_1 = LogisticRegression(random_state=0).fit(train_x_1, ravel(train_y_1))
predict_all_1 = clf_all_1.predict(test_x)
print(np.mean(predict_all_1==ravel(test_y)))
#Profit
kk_1 = test_k
for i in range(0,len(predict_all_1)):
       if predict_all_1[i-1]==1:
           kk_1[i:i+1] = 0
print(sum(kk_1['PROFIT_LOSS']))


#Client 2
train_2 = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_2.csv")
train_y_2 = pd.DataFrame(train_2, columns=col_n_y)
train_x_2 = pd.DataFrame(train_2, columns=col_n_x)
#model 2
#Default
clf_all_2 = LogisticRegression(random_state=0).fit(train_x_2, ravel(train_y_2))
predict_all_2 = clf_all_2.predict(test_x)
print(np.mean(predict_all_2==ravel(test_y)))
#Profit
kk_2 = test_k
for i in range(0,len(predict_all_2)):
       if predict_all_2[i-1]==1:
           kk_2[i:i+1] = 0
print(sum(kk_2['PROFIT_LOSS']))

#Client 3
train_3 = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_3.csv")
train_y_3 = pd.DataFrame(train_3, columns=col_n_y)
train_x_3 = pd.DataFrame(train_3, columns=col_n_x)
#model 3
#Default
clf_all_3 = LogisticRegression(random_state=0).fit(train_x_3, ravel(train_y_3))
predict_all_3 = clf_all_3.predict(test_x)
print(np.mean(predict_all_3==ravel(test_y)))
#Profit
kk_3 = test_k
for i in range(0,len(predict_all_3)):
       if predict_all_3[i-1]==1:
           kk_3[i:i+1] = 0
print(sum(kk_3['PROFIT_LOSS']))

#Clien 4
train_4 = pd.read_csv("/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_4.csv")
train_y_4 = pd.DataFrame(train_4, columns=col_n_y)
train_x_4 = pd.DataFrame(train_4, columns=col_n_x)
#model 4
#Default
clf_all_4 = LogisticRegression(random_state=0).fit(train_x_4, ravel(train_y_4))
predict_all_4 = clf_all_4.predict(test_x)
print(np.mean(predict_all_4==ravel(test_y)))
#Profit
kk_4 = test_k
for i in range(0,len(predict_all_4)):
       if predict_all_4[i-1]==1:
           kk_4[i:i+1] = 0
print(sum(kk_4['PROFIT_LOSS']))




# 0.6932666666666667
# 107960
#
#
# #0.6872333333333334 profit: 106190
#
# 0.6452333333333333
# 93570
#
# 0.6457
# 93280
#
# 0.6436666666666667
# 92900
#
# 0.6782333333333334
# 90100
#
# INFO flower 2021-11-27 17:30:45,164 | app.py:122 | app_fit: metrics_centralized {'accuracy': [(0, 0.6666666666666666), (1, 0.6666666666666666), (2, 0.3438333333333333), (3, 0.6586333333333333), (4, 0.6835333333333333), (5, 0.6735666666666666), (6, 0.6818666666666666), (7, 0.6764), (8, 0.6608), (9, 0.6822), (10, 0.40116666666666667), (11, 0.6839), (12, 0.6656666666666666), (13, 0.6872333333333334), (14, 0.4046), (15, 0.5139666666666667), (16, 0.6570666666666667), (17, 0.6325666666666667), (18, 0.3624), (19, 0.6279666666666667), (20, 0.5032)]}
# INFO flower 2021-11-27 17:34:50,643 | app.py:122 | app_fit: metrics_centralized {'accuracy': [(0, 'accuracy: 0.6666666666666666 profit: 100000'), (1, 'accuracy: 0.6666666666666666 profit: 100000'), (2, 'accuracy: 0.6549333333333334 profit: 96480'), (3, 'accuracy: 0.6765666666666666 profit: 102970'), (4, 'accuracy: 0.6763 profit: 102890'), (5, 'accuracy: 0.6793666666666667 profit: 103810'), (6, 'accuracy: 0.6818333333333333 profit: 104550'), (7, 'accuracy: 0.48033333333333333 profit: 44100'), (8, 'accuracy: 0.5746333333333333 profit: 72410'), (9, 'accuracy: 0.3742 profit: 12260'), (10, 'accuracy: 0.6029666666666667 profit: 80910'), (11, 'accuracy: 0.5531333333333334 profit: 65960'), (12, 'accuracy: 0.38503333333333334 profit: 15510'), (13, 'accuracy: 0.5692666666666667 profit: 70800'), (14, 'accuracy: 0.5808333333333333 profit: 74270'), (15, 'accuracy: 0.6836333333333333 profit: 105110'), (16, 'accuracy: 0.67 profit: 101020'), (17, 'accuracy: 0.6853 profit: 105610'), (18, 'accuracy: 0.6678 profit: 100340'), (19, 'accuracy: 0.6788666666666666 profit: 103680'), (20, 'accuracy: 0.6889666666666666 profit: 106710')]}
# accuracy: 0.6889666666666666 profit: 106710





