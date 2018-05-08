# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('E:/Python Github/Datasets/Titanic/train.csv')
test = pd.read_csv('E:/Python Github/Datasets/Titanic/test.csv')
full_data = [train, test]

print(train.columns.values)
train.head()

train.info()
print('_'*40)
test.info()

train.describe()
train.describe(include=['O'])

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train)

for dataset in full_data:# Mapping Gender
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)   