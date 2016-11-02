# builtin, remove warnings (different from exceptions)
# can turn warnings to exceptions by using 'error' as the parameter
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("C:\\Users\\Jacob\\Anaconda3\\Lib\\site-packages")

""" Base and visualization modules """
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

""" Model evaulators """
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn_ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_Validation import cross_val_score

""" Feature importance estimators """
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# targetdata is the "Survival" column
##data = pd.read_csv('train.csv')
##
### replace NaN values in age with the median (more robust than mean against
### outliers
##data['Age'].fillna(data['Age'].median(), inplace=True)

# ____ visualization chart of gender ____
##survived_sex = data[data['Survived']==1]['Sex'].value_counts()
##dead_sex = data[data['Survived']==0]['Sex'].value_counts()
##df_sex = pd.DataFrame([survived_sex, dead_sex])
##df_sex.index = ['Survived','Dead']
##df_sex.plot(kind='bar', stacked=True, figsize=(15,8))
##plt.show()

""" shows that men make up a larger portion of people who died and a
smaller portion of people who lived """

# ____ visualization chart of age ____
##plt.figure(figsize=(7,4))
##plt.hist([data[data["Survived"]==1]["Age"],data[data["Survived"]==0]["Age"]] \
##         ,bins=30, stacked=True, color=['g','r'], label=["Survived","Dead"])
##plt.xlabel("Age")
##plt.ylabel("#")
##plt.legend()
##plt.show()

""" Children more likeley to survive, large peak at the artificial median age
that replaced NaN suggests that replacing missing values with median was not
a good approx. of actual age distribution """

# _____ visualization of fare ticket price on survival _____
##plt.figure(figsize=(7,4))
##plt.hist([data[data["Survived"]==1]["Fare"],data[data["Survived"]==0]["Fare"]],
##         bins=30,stacked=True, color=['g','r'],label=["Survied","Dead"])
##plt.xlabel("Fare price")
##plt.ylabel("#")
##plt.legend()
##plt.show()

""" Fare prices above 100 are a strong predictor of survival, almost all green
below are only about 50% """

# ___ multi element (age, fare) visualization _______
##plt.figure(figsize=(7,4))
##ax = plt.subplot()
##ax.scatter(data[data["Survived"]==1]["Age"],data[data["Survived"]==1]["Fare"],c="green",s=40)
##ax.scatter(data[data["Survived"]==0]["Age"],data[data["Survived"]==0]["Fare"],c="red",s=40)
##ax.set_xlabel("Age")
##ax.set_ylabel("Fare")
##ax.legend(("Survived","Dead"),scatterpoints=1,loc="upper right", fontsize=10)
##plt.show()

""" Fare price is the larger factor affecting survival """

# ____ visualization of embarkment point and survival ______
##survived_embark = data[data["Survived"]==1]["Embarked"].value_counts()
##dead_embark = data[data["Survived"]==0]["Embarked"].value_counts()
##df = pd.DataFrame([survived_embark,dead_embark])
##df.index = ["Survived","Dead"]
##df.plot(kind="bar", stacked=True, figsize=(7,4))
##plt.show()

""" No clear differnce, distribution of embarkment points looks same"""

# function that asserts whether a feature has been processed
def status(feature):
    print("Processing",feature,": ok")

# combine test and train into a single dataframe so that feature munging is
# done on both at once, will break apart prior to training
def get_combined_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    targets = train.Survived
    train.drop("Survived",1,inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop("index",inplace=True,axis=1)
    return combined

def get_titles(dataframe,column):
    dataframe["Title"] = combined[column].map(lambda x: x.split()[1]\
                                             .split('.')[0].strip())
    Title_Dictionary = {
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
    dataframe['Title'] = combined.Title.map(Title_Dictionary)
def process_age():
    global combined
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    combined.Age = combined.apply(lambda r : fillAges(r)
                                  if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')

def process_names():
    global combined
    combined.drop("Name",axis=1,inplace=True)
    titles_dummies = pd.get_dummies(combined["Title"], prefix="Title")
    combined = pd.concat([combined,titles_dummies],axis=1)
    combined.drop("Title",axis=1,inplace=True)
    status('names')
def process_fares():
    global combined
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    status("fare")
def process_embarked():
    global combined
    combined.Embarked.fillna('S',inplace=True)
    embarked_dummies = pd.get_dummies(combined["Embarked"],prefix="Embarked")
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop("Embarked",axis=1,inplace=True)
    status('embarked')
def process_cabin():
    global combined
    combined.Cabin.fillna("U",inplace=True)
    combined["Cabin"] = combined["Cabin"].map(lambda x: x[0])
    cabin_dummies = pd.get_dummies(combined["Cabin"],prefix="Cabin")
    combined = pd.concat([combined,cabin_dummies],axis=1)
    combined.drop("Cabin",axis=1,inplace=True)
    status('cabin')
def process_sex():
    global combined
    combined["Sex"] = combined["Sex"].map({'male':1,'female':0})
    status('sex')
def process_pclass():
    global combined
    pclass_dummies = pd.get_dummies(combined["Pclass"],prefix="class")
    combined = pd.concat([combined,pclass_dummies],axis=1)
    combined.drop("Pclass",axis=1,inplace=True)
    status('pclass')
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')

def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    
    status('family')

# feature engineering using functions above
combined=get_combined_data()
get_titles(combined, "Name")
process_age()
process_names()
process_fares()
process_embarked()
process_cabin()
process_sex()
process_pclass()
process_ticket()
process_family()

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)
def recover_train_test():
    global combined
    train0 = pd.read('train.csv')
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    return train,test,targests
train,test,targets = recover_train_test()

# estimating feature importance
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)

