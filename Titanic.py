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

#read data
train = pd.read_csv('E:/Python Github/Datasets/Titanic/train.csv')
test = pd.read_csv('E:/Python Github/Datasets/Titanic/test.csv')
full = train.append( test , ignore_index = True )

full.describe()
full.info()

#fillna
full['Age']=full['Age'].fillna( full['Age'].mean() )
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )
full.info()

full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].fillna( 'S' )

#U=Unknown
full['Cabin'] = full['Cabin'].fillna( 'U' )
full.info()

#One-hot
sex_mapDict={'male':1,'female':0}
full['Sex']=full['Sex'].map(sex_mapDict)

embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
embarkedDf.head()
full = pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
full.head()

pclassDf = pd.DataFrame()
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)
full.head()

def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

titleDf = pd.DataFrame()
titleDf['Title'] = full['Name'].map(getTitle)
titleDf.head()
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
titleDf = pd.get_dummies(titleDf['Title'])
titleDf.head()
full = pd.concat([full,titleDf],axis=1)
full.drop('Name',axis=1,inplace=True)

cabinDf = pd.DataFrame()
full[ 'Cabin' ] = full[ 'Cabin' ].map( lambda c : c[0] )
cabinDf = pd.get_dummies( full['Cabin'] , prefix = 'Cabin' )
full = pd.concat([full,cabinDf],axis=1)
full.drop('Cabin',axis=1,inplace=True)
full.head()

familyDf = pd.DataFrame()
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
full.head()

#correlation
corrDf = full.corr() 
corrDf
corrDf['Survived'].sort_values(ascending =False)

full_X = pd.concat( [titleDf,
                     pclassDf,
                     familyDf,
                     full['Fare'],
                     cabinDf,
                     embarkedDf,
                     full['Sex']
                    ] , axis=1 )

#model fitting
sourceRow=891
source_X = full_X.loc[0:sourceRow-1,:]
source_y = full.loc[0:sourceRow-1,'Survived']  
pred_X = full_X.loc[sourceRow:,:]
source_X.shape[0]

from sklearn.cross_validation import train_test_split 
train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                    source_y,
                                                    train_size=.8)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit( train_X , train_y )
model.score(test_X , test_y )
print(model.score(test_X , test_y ))

pred_Y = model.predict(pred_X)
pred_Y=pred_Y.astype(int)

passenger_id = full.loc[sourceRow:,'PassengerId']
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y } )
predDf.shape
predDf.head()
predDf.to_csv( 'titanic_pred.csv' , index = False )

#Trying another model
model2 = LogisticRegression()
model2.fit( source_X , source_y )

pred_Y2 = model2.predict(pred_X)
pred_Y2=pred_Y2.astype(int)

passenger_id = full.loc[sourceRow:,'PassengerId']
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y2 } )
predDf.shape
predDf.head()
predDf.to_csv( 'titanic_pred2.csv' , index = False )














