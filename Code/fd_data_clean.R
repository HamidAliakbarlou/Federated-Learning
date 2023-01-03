#Assignment 1 : Credit Risk Game
#@author : Rongde Liang
#@version: 2021-10-14
#@Description: the code herein was used to produce the prediction of 'CreditGame_Applications.csv'
#              
#              Prediction Model: Logistic Regression
#
#              Candidate : Model 1 full model
#                          Model 2 stepwise selection from the full model
#                          Model 3 full model with interactions and quadratic powers
#                          Model 4 stepwise selection from the full model with interactions and quadratic powers
#
#              Final Choice: Model 3 with 15 interactions and quadratic powers
#                            **** full model3 and model4 cause vector memory exhausted on my computer
#
#              Catalogue: 1. Prerequistite
#                         2. Exploring the data
#                         3. Cleaning the data
#                         4. Modeling the data
#                         5. Predicting the data
#                         **** for the sale of reivewing the model,please check 'Modeling the data' Model3 directly


######0. Prerequistite######
#Library
library(dplyr)

#Source
source("/Users/rongliang/Downloads/Week5-PredBinaries/PredictingBinaries.R")

#Data
#Training data
full_train_data = read.csv("/Users/rongliang/Documents/Course/5.Statistical_Learning/Credit\ Game/CreditGameData2021/CreditGame_TRAIN.csv")
set.seed(20211012)

full_train_data1 = full_train_data %>% select(ID_TRAIN,ST_EMPL,NB_EMPT,AGE_D,DEFAULT,PROFIT_LOSS,REV_BT,REV_NET,MNT_EPAR,NB_ER_6MS,NB_ER_12MS,NB_DEC_12MS,NB_COUR,NB_INTR_1M,NB_INTR_12M,NB_DEL_30,MNT_PASS,MNT_DEMANDE,everything())


#Testing data
test_data = read.csv("/Users/rongliang/Documents/Course/5.Statistical_Learning/Credit\ Game/CreditGameData2021/CreditGame_Applications.csv")

######1.Exploring the data######
#data size
full_train_data %>% nrow() # 1000000

#display the internal structure
str(full_train_data) #TYP_RES,ST_EMPL,TYP_FIN are factor

#head data
head(full_train_data)

#summary data
summary(full_train_data)

#summary miss data
summary(full_train_data[,'AGE_D'])  #median is 34
summary(as.factor(full_train_data[,'ST_EMPL']))  #most frequency is 'R'

#######2.Clean Data############
# replace missing value AGE_D with the median value 34
full_train_data_1= full_train_data %>%
  filter(is.na(AGE_D)) %>%
  mutate(AGE_D=34)

full_train_data_2= full_train_data %>%
  filter(!is.na(AGE_D))

full_train_data = rbind(full_train_data_1,full_train_data_2)

# replace missing value ST_EMPL with most frequency 'R'
full_train_data_1= full_train_data %>%
  filter(ST_EMPL=='') %>%
  mutate(ST_EMPL='R')

full_train_data_2= full_train_data %>%
  filter(ST_EMPL!='')

full_train_data = rbind(full_train_data_1,full_train_data_2)

# transformation: TYP_RES P: Owner 0, L: Tenant 1, A: Others 2
full_train_data_1= full_train_data %>%
  filter(TYP_RES=='P') %>%
  mutate(TYP_RES='0')

full_train_data_2= full_train_data %>%
  filter(TYP_RES=='L') %>%
  mutate(TYP_RES='1')

full_train_data_3= full_train_data %>%
  filter(TYP_RES=='A') %>%
  mutate(TYP_RES='2')

full_train_data = rbind(full_train_data_1,full_train_data_2,full_train_data_3)

# transformation: ST_EMPL:R: Regular 0, P: Part-Time 1, T: Self Employed 2

full_train_data_1= full_train_data %>%
  filter(ST_EMPL=='R') %>%
  mutate(ST_EMPL='0')

full_train_data_2= full_train_data %>%
  filter(ST_EMPL=='P') %>%
  mutate(ST_EMPL='1')

full_train_data_3= full_train_data %>%
  filter(ST_EMPL=='T') %>%
  mutate(ST_EMPL='2')

full_train_data = rbind(full_train_data_1,full_train_data_2,full_train_data_3)


# cleaning data for fd algorithm
#sample
k = (full_train_data %>% filter(DEFAULT==0))[1:100000,]
k = k %>% filter(DEFAULT==0) %>% mutate(PROFIT_LOSS=10)
g = (full_train_data %>% filter(DEFAULT==1))[1:50000,]
g = g %>% filter(DEFAULT==1) %>% mutate(PROFIT_LOSS=-10)

kl = (full_train_data %>% filter(DEFAULT==0))[1000001:120000,]
kl = k %>% filter(DEFAULT==0) %>% mutate(PROFIT_LOSS=10)
gl = (full_train_data %>% filter(DEFAULT==1))[50000:56000,]
gl = g %>% filter(DEFAULT==1) %>% mutate(PROFIT_LOSS=-10)

#training data
set.seed(20211126)
trainID_1=sample(1:nrow(k),nrow(k)*0.8)
train_1=k[trainID_1,]
trainID_2=sample(1:nrow(g),nrow(g)*0.8)
train_2=g[trainID_2,]

train = rbind(train_1,train_2)

#validation data
validation = rbind(kl,gl)

#test data
test_1=k[-trainID_1,]
test_2=g[-trainID_2,]

test = rbind(validate_1,validate_2)

#write
write.table(train,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_all.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(validation,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_VALIDATION_all_test.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(test,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TEST_all.csv",row.names=FALSE,col.names=TRUE,sep=",")

#sum(validate[,'PROFIT_LOSS'])

#client
#client 1
set.seed(202111261)
trainID_1=sample(1:nrow(train_1),nrow(train_1)*0.25)
train_1_1=k[trainID_1,]

trainID_2=sample(1:nrow(train_2),nrow(train_2)*0.5)
train_2_1=g[trainID_2,]

train_1_1_1 = rbind(train_1_1,train_2_1)
write.table(train_1_1_1,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_1.csv",row.names=FALSE,col.names=TRUE,sep=",")

#client 2
set.seed(202111127)
trainID_1=sample(1:nrow(train_1),nrow(train_1)*0.25)
train_1_1=k[trainID_1,]

trainID_2=sample(1:nrow(train_2),nrow(train_2)*0.5)
train_2_1=g[trainID_2,]

train_1_1_2 = rbind(train_1_1,train_2_1)
write.table(train_1_1_2,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_2.csv",row.names=FALSE,col.names=TRUE,sep=",")

#client 3
set.seed(203339)
trainID_1=sample(1:nrow(train_1),nrow(train_1)*0.25)
train_1_1=k[trainID_1,]

trainID_2=sample(1:nrow(train_2),nrow(train_2)*0.5)
train_2_1=g[trainID_2,]

train_1_1_3 = rbind(train_1_1,train_2_1)
write.table(train_1_1_3,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_3.csv",row.names=FALSE,col.names=TRUE,sep=",")

#client 4
set.seed(2021112655)
trainID_1=sample(1:nrow(train_1),nrow(train_1)*0.25)
train_1_1=k[trainID_1,]

trainID_2=sample(1:nrow(train_2),nrow(train_2)*0.5)
train_2_1=g[trainID_2,]

train_1_1_4 = rbind(train_1_1,train_2_1)
write.table(train_1_1_4,"/Users/rongliang/PycharmProjects/untitled7/credit/CreditGame_TRAIN_4.csv",row.names=FALSE,col.names=TRUE,sep=",")

#plot
round <- c(1:21)
accuracy <- c(0.6666, 0.6666, 0.6549, 0.6765, 0.6763, 0.6793, 0.6818, 0.4803, 0.5746, 0.3742, 0.6029, 0.5531, 0.3850, 0.5692, 0.5808, 0.6836, 0.67, 0.6853, 0.6678, 0.6788,  0.6889)
profit <- c(100000, 100000, 96480, 102970, 102890, 103810, 104550, 44100, 72410, 12260, 80910, 65960, 15510, 70800, 74270, 105110, 101020, 105610, 100340, 103680,  106710)


round <- c(0.01,0.05,0.1,0.5 ,1,5,10)
round <- c(1:7)
accuracy <- c(0.6644, 0.67, 0.6675, 0.6803, 0.6889,0.6198,0.5511)
profit <- c(99330, 101000, 100270, 104090, 106710, 85960, 65340)


ra <-  rbind(round,accuracy)

plot(x=round, y=accuracy, sub="Accuracy", type="l",xlab="C",ylab="Accuracy",ylim=c(0.50,0.70),xaxt="n")
axis(side=1,at=c(1:7),labels=c(0.01,0.05,0.1,0.5 ,1,5,10))


plot(x=round, y=profit, sub="Porift", type="l",xlab="C",ylab="Profit",ylim=c(60000,110000),xaxt="n")
axis(side=1,at=c(1:7),labels=c(0.01,0.05,0.1,0.5 ,1,5,10))






