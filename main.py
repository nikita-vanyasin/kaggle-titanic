import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from helpers import DataFrameImputer

df_src = pd.read_csv('data/train.csv')
df_questioned_src = pd.read_csv('data/test.csv')

sex_mapping = {'male': 0, 'female': 1}
embarked_mapping = {'Q': 0, 'C': 1, 'S': 2}


def drop_cols(df):
    # TODO: why we assumed that cabin number not affected final result?
    cols = ['PassengerId', 'Cabin', 'Name', 'Ticket', 'Embarked']
    for col in cols:
        df = df.drop(col, axis=1)
    return df


def map_sex(df):
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df


def map_embarked(df):
   # df['Embarked'] = df['Embarked'].map(embarked_mapping)
    return df


def preprocessing_pipeline(df):
    # clean
    df = drop_cols(df)

    # fill or remove empty values
    df = DataFrameImputer().fit_transform(df)

    # convert categorical features to numbers
    df = map_sex(df)
    df = map_embarked(df)

    return df


df_src = preprocessing_pipeline(df_src)
df_questioned = preprocessing_pipeline(df_questioned_src)

# extract answers
df_src_survived = df_src.loc[:, 'Survived']
df_src = df_src.drop(['Survived'], axis=1)

# encode categorical features
ce_encoder = ce.OneHotEncoder(cols=['Pclass', 'SibSp', 'Sex', 'Parch'], drop_invariant=True, impute_missing=False)
ce_encoder.fit(df_src)
df_src = ce_encoder.transform(df_src)
df_questioned = ce_encoder.transform(df_questioned)

column_info = df_src.columns

# split src data for testing later
X_train, X_test, y_train, y_test = train_test_split(df_src, df_src_survived, test_size=0.05, random_state=14)

# scale
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train = stdsc.transform(X_train)
X_test = stdsc.transform(X_test)
X_questioned = stdsc.transform(df_questioned)


# skipped step:
# TODO: may be we need to choose meaningful fields using regularization or do PCA
# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)


# train models
def run_and_check(name, classifier, X_train, y_train, X_test, y_test):
    classifier = classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    #print("%.3f" % accuracy_score(y_train, y_train_pred))
    y_test_pred = classifier.predict(X_test)
    print(name + ":\t%.3f/%.3f" % (accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)))


run_and_check('knbr', KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski', algorithm='brute'), X_train, y_train, X_test, y_test)

tree = DecisionTreeClassifier(criterion='gini', random_state=10)
run_and_check('tree', tree, X_train, y_train, X_test, y_test)

ada_tree = AdaBoostClassifier(base_estimator=tree, n_estimators=1000, learning_rate=0.05, random_state=10)
run_and_check('ada_tree', ada_tree, X_train, y_train, X_test, y_test)

forest = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=10)
run_and_check('forest', forest, X_train, y_train, X_test, y_test)


# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
# shape = X_train.shape[1]
# for f in range(shape):
#     print("%2d) %-*s %f" % (f + 1, 30, column_info[indices[f]], importances[indices[f]]))

# pcn = Perceptron(penalty='l2', alpha=0.0001, n_iter=50, shuffle=True)
# run_and_check('pcn', pcn, X_train, y_train, X_test, y_test)
#
# sgd = SGDClassifier(n_iter=500, loss='perceptron')
# run_and_check('sgd', sgd, X_train, y_train, X_test, y_test)
#
# mlp = MLPClassifier(hidden_layer_sizes=(50, 50))
# run_and_check('mlp', mlp, X_train, y_train, X_test, y_test)

Y_questioned = ada_tree.predict(X_questioned)


df_result = pd.DataFrame({'PassengerId': df_questioned_src.loc[:, 'PassengerId'], 'Survived': Y_questioned})

# print(df_result)

df_result.to_csv('data/result.csv', index=False)