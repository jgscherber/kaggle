import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier

## ____________________ Set Up Training Data ____________________
titanic_df = pd.read_csv('train.csv', header=0)

#titanic_df["Gender"] = titanic_df["Sex"].map(lambda x: x[0].upper())
titanic_df["Gender"] = titanic_df["Sex"].map({'female':0,'male':1}).astype(int)

median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = titanic_df[(titanic_df["Gender"] == i) &
                                      (titanic_df["Pclass"] == j+1)]["Age"]\
                                      .dropna().median()

titanic_df["AgeFill"] = titanic_df["Age"]
##for i in titanic_df[titanic_df["AgeFill"].isnull()]:
##    titanic_df["Age"] = median_ages[titanic_df["Gender"]][titanic_df["Pclass"]]
for i in range(0,2):
    for j in range(0,3):
        # loc can be used with a boolean array
        titanic_df.loc[(titanic_df.Age.isnull()) & (titanic_df.Gender == i), \
                       "AgeFill"] = median_ages[i,j]
titanic_df["AgeIsNull"] = pd.isnull(titanic_df["Age"]).astype(int)
titanic_df["FamilySize"] = titanic_df["SibSp"] + titanic_df["Parch"]

# Artifical feature that will accentuate Pclass 3 and older ages, both who were
# unlikely to survive
titanic_df["Age*Class"] = titanic_df.AgeFill * titanic_df.Pclass

# drop columns that aren't numerical and columns missing values
titanic_df = titanic_df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked" \
                              ,"Age", "PassengerId"], axis=1)

# Turn the panda dataframe into a numpy array
train_data = titanic_df.values

## ____________________ Setup Test Data ___________________
test_file = open("test.csv",'rb')
test_array = csv.reader(test_file)
header = test_array.next()
# windows files have weird way of opening and saving some files that they can
# append a different endline indicator than unix systems. wb, rb, etc. open it
# as a binary file, allowing interoperabilty with unix (binary open also works
# on unix so code is cross-platform )
prediction_file = open("randomforestmodel.csv",'wb')
prediction_array = csv.writer(prediction_file)
# n_estimators is the number of trees in the forest
forest = RandomForestClassifier(n_estimators = 100)
# X = input data (all non-survival columns
# Y = survival data
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
output = forest.predict(test_data)

