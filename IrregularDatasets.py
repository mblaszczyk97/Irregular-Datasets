import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import imblearn
import itertools
from collections import Counter
from matplotlib import pyplot
from numpy import where

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier


def logistic_regression(X_train, y_train, X_test, y_test, X_train_res, y_train_res):
    parameters = {
    'C': np.linspace(1, 10, 10)
             }
    lr = LogisticRegression()
    lr_without = LogisticRegression()
    #clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
    #clf.fit(X_train_res, y_train_res.ravel())
    lr2 = LogisticRegression(C=1.0,penalty='l2', verbose=5)
    lr2.fit(X_train_res, y_train_res.ravel())
    lr_without.fit(X_train, y_train.values.ravel())

    y_train_pre_without = lr_without.predict(X_test)
    cnf_matrix_tra_without = confusion_matrix(y_test, y_train_pre_without)
    without=100*cnf_matrix_tra_without[1,1]/(cnf_matrix_tra_without[1,0]+cnf_matrix_tra_without[1,1])
    print("Funkcja regresjii logistycznej (Bez oversamplingu): {}%".format(without))
    print(cnf_matrix_tra_without[0,0],cnf_matrix_tra_without[1,1])


    y_train_pre = lr2.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_pre)
    within=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Funkcja regresjii logistycznej (Z oversamplingiem): {}%".format(within))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    objects = ('Regresja Logistyczna', 'Regresja Logistyczna z oversamplingiem')
    y_pos = np.arange(len(objects))
    performance = [without, within]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Procent dokładności')
    plt.title('Dokładność Regresjii Logistycznej')
    plt.show()
    return without, within

def random_forest(X_train, y_train, X_test, y_test, X_train_res, y_train_res):
    rf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train.values.ravel())
    y_train_rf = rf.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_rf)
    without=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Random Forest (niezbalansowany): {}%".format(without))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    rf_oversampling = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    rf_oversampling.fit(X_train_res, y_train_res.ravel())
    y_train_rf = rf_oversampling.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_rf)
    with_oversampling=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Random Forest (z oversamplingiem): {}%".format(without))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    brf = BalancedRandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    brf.fit(X_train, y_train.values.ravel())
    y_train_brf = brf.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_brf)
    within=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Random Forest (zbalansowany - undersampling): {}%".format(within))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])
    print(brf.feature_importances_)
    
    objects = ('country','gender', 'age', 'visiting Wuhan', 'from Wuhan')
    y_pos = np.arange(len(objects))
    performance = brf.feature_importances_*100
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Procent zależności')
    plt.title('Zależność poszczególnych atrybutów')
    plt.show()

    objects = ('Random Forest niezbalansowany','Random Forest z oversamplingiem', 'Random Forest zbalansowany')
    y_pos = np.arange(len(objects))
    performance = [without, with_oversampling, within]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Procent dokładności')
    plt.title('Dokładność Random Forest')
    plt.show()

    return without, within

def adaboost(X_train, y_train, X_test, y_test):
    base_estimator = AdaBoostClassifier(n_estimators=10)
    eec = EasyEnsembleClassifier(n_estimators=10, base_estimator=base_estimator, n_jobs=-1)
    eec.fit(X_train, y_train.values.ravel())
    y_train_eec = eec.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_eec)
    without=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Adaboost (boosting): {}%".format(without))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    objects = ('Boosting', '-')
    y_pos = np.arange(len(objects))
    performance = [without, 0]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Procent dokładności')
    plt.title('Dokładność Adaboost z losowym undersamplingiem')
    plt.show()

    return without



def balanced_bragging(X_train, y_train, X_test, y_test, X_train_res, y_train_res):
    bagging = BaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    bagging.fit(X_train, y_train.values.ravel())
    y_train_bc = bagging.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_bc)
    without=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Niezbalansowane (bragging): {}%".format(without))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    bagging_oversampling = BaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    bagging_oversampling.fit(X_train_res, y_train_res.ravel())
    y_train_bc = bagging_oversampling.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_bc)
    with_oversampling=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("z oversamplingiem (bragging): {}%".format(with_oversampling))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    balanced_bagging = BalancedBaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    balanced_bagging.fit(X_train, y_train.values.ravel())
    y_train_bbc = balanced_bagging.predict(X_test)
    cnf_matrix_tra = confusion_matrix(y_test, y_train_bbc)
    within=100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
    print("Zbalansowane (bragging): {}%".format(within))
    print(cnf_matrix_tra[0,0],cnf_matrix_tra[1,1])

    objects = ('Bragging','Bragging z oversamplingiem SMOTE', 'Bragging z losowym undersamplingiem')
    y_pos = np.arange(len(objects))
    performance = [without,with_oversampling, within]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Procent dokładności')
    plt.title('Dokładność braggingu')
    plt.show()
    return without, within

def graph(df):
    target_count = df.target.value_counts()
    print('Survive:', target_count[0])
    print('Death:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

    pd.value_counts(df['target']).plot.bar()
    plt.title('Corona Virus death vs survivors')
    plt.xlabel('Number of deaths')
    plt.ylabel('Frequency')
    plt.show()
    df['target'].value_counts()

#READ FILES BEGIN
df = pd.read_csv('../input/coronaVirusCases.csv')
#READ FILES END

labelencoder = LabelEncoder()

y = df[['id', 'target']].set_index('id')
X = df.drop(['target'], axis=1).set_index('id')
X['country'] = labelencoder.fit_transform(X['country'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Stosunek smierci do uzdrowien przed funkcją SMOTE")
countedDeaths = y_train.pivot_table(index=['target'], aggfunc='size')
print (countedDeaths)

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.values.ravel())

print("Stosunek smierci do uzdrowien po funkcji SMOTE")
print("0: ", sum(y_train_res==0))
print("1: ", sum(y_train_res==1))

graph(df)
#logistic_regression(X_train, y_train, X_test, y_test, X_train_res, y_train_res)
#balanced_bragging(X_train, y_train, X_test, y_test, X_train_res, y_train_res)
#adaboost(X_train, y_train, X_test, y_test)
#random_forest(X_train, y_train, X_test, y_test, X_train_res, y_train_res)