import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/train.csv").replace("male", 0).replace("female", 1)

df["Age"].fillna(df.Age.median(), inplace = True)

df = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis = 1)

train_data = df.values
xs = train_data[:, 2:]
y = train_data[:, 1]

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(xs, y)

test_df = pd.read_csv("data/test.csv").replace("male", 0).replace("female", 1)
test_df["Age"].fillna(df.Age.median(), inplace = True)
test_df = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis = 1)

test_data = test_df.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

with open("data/result.csv", "w") as f:
    writer = csv.writer(f, lineterminator = "\n")
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:, 0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
