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

train_df = pd.read_csv('E:/Python Github/Datasets/Titanic/train.csv')
test_df = pd.read_csv('E:/Python Github/Datasets/Titanic/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)
train_df.head()

train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
train_df.describe(include=['O'])

