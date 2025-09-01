import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

#load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# print(train_df.isnull().sum())
# print(test_df.isnull().sum())

#data cleaning
train_df["Age"] = train_df["Age"].fillna(train_df['Age'].median())
test_df["Age"] = test_df["Age"].fillna(test_df['Age'].median())

test_passenger_ids = test_df["PassengerId"]

train_df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)
test_df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)

train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
#encode categorial data
le = LabelEncoder() #create lable encoder instance

#Encode 'sex'
train_df["Sex"] = le.fit_transform(train_df["Sex"])
test_df["Sex"] = le.transform(test_df["Sex"])

#Encode 'Embarked'
train_df["Embarked"] = le.fit_transform(train_df["Embarked"])
test_df["Embarked"] = le.transform(test_df["Embarked"])

#train the model logistic regression

#split features and target
x = train_df.drop("Survived",axis=1)
y = train_df["Survived"]

#Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(x,y)

#make predictions on the test set
predictions = model.predict(test_df)

#prepare the submission file
submission = pd.DataFrame({
    "PassengerId" : test_passenger_ids,
    "Pclass" : test_df["Pclass"],
    "Age" : test_df["Age"],
    "Survived": predictions
})

submission.to_csv("submission.csv",index=False)
print("submission.csv created successfully.")
