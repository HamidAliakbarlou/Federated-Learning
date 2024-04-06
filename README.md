# Application of federated learning to a credit risk model


In this project, we successfully explored an application of federated learning to a credit risk model for loan approval, and compare its accuracy to that of a traditional, centralized modelling approach
The challenge was to obtain a model through decentralized computation that is comparable to a benchmark model obtained through centralized means. We used FedAvg, and the Flower framework for federated learning to learn the shared weights in a logistic regression model. In the end, the centralized model was 69.32% accurate, while the FL model was 69.23% accurate.

## Dataset
The considered dataset takes into account an applicant’s financial profile, made up of features such as loan amount, age, income (net and gross), value of savings, transactional records, assets, liabilities and more, to determine their likelihood of default. In total, we are given 27 descriptive features and one target variable, which was an indicator dummy telling us whether a default had occurred. The dataset was simulated by a professional with over 20 years of experience in financial modelling, and is meant to closely approximate what is observed in the real world.

### Data File Structure
There are two dataset in a “credit” file folder: CreditGame_TRAIN_all.csv is the training dataset; CreditGame_ TEST_all.csv is the testing dataset. In our FL setup, the server component utilizes the whole training dataset and testing dataset while each client uses a random sample of one quarter of both datasets. Therefore, the data is partitioned evenly between clients. Our sampling method does this automatically when we run the code.

### Code File Structure
There are 7 python files in our project, described in Table 1 available in the "Report" file. The coding component comprises three parts: the client scripts, the server script and the utils code. The utils code must be put together with client code or server code since it contains the command parameters and utility functions. The server code and client code can be run in parallel on different machines.
