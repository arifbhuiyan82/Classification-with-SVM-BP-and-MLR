Classification with SVM, BP and MLR

Assignemt done by Arif Bhuiyan & Julio Cesar

Objective: Perform data classification with the following algorithms:

- Support Vector Machines (SVM), using free software
- Back-Propagation (BP), using free software
- Multiple Linear Regression (MLR), using free software

The classification must be performed on three datasets:

Ring datasets (A2-ring.zip):
- Training set1: ring-separable.txt
- Training set2: ring-merged.txt
- Test (valid for set1 and set2): ring-test.txt
- 2 input features + 1 class identifier (0 / 1)
- Two different training sets, one easy (separable) and one more difficult (merged)
- Only one test set for both training sets
- All data files have 10000 patterns
- Since there are only two input features, plot the data to know the meaning of this classification task

Bank marketing dataset (A2-bank.zip):
- Data: bank-additional.csv (4119 patterns) or bank-additional-full.csv (41188 patterns), choose one of them (the first is a subset of the second)
- Training: select the first 80% patterns for training
- Test: select the last 20% patterns for test
- Features: 20 features, most of them categorical, you will have to properly represent them as numerical data before training
- Input features: features that refer to the bank client, last contact in the current campaign, other attributes, and social and economic context attributes
- Prediction feature: the last one (yes/no), which corresponds to whether the client has subscribed a term deposit or not
- Observation: missing information is tagged as “unknown”


Search a dataset from the Internet, with the following characteristics: (we use Prime Indians Diabetes Dataset downloaded from https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
- At least 6 features, one of them used for classification
- The classification feature can be binary or multivariate
- At least 400 patterns
- Select randomly 80% of the patterns for training and validation, and the remaining 20% for test; it is important to shuffle the original data, to destroy any kind of sorting it could have

Part1: Selecting and analyzing the datasets
Part 2: Classification problem
	- Parameter selection
	- Evaluation of the results