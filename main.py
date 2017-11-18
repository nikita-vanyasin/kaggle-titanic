import category_encoders as ce
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from helpers import DataFrameImputer

df_src = pd.read_csv('data/train.csv')
df_questioned = pd.read_csv('data/test.csv')

sex_mapping = {'male': 0, 'female': 1}
embarked_mapping = {'Q': 0, 'C': 1, 'S': 2}


def drop_cols(df):
    # TODO: why we assumed that cabin number not affected final result?
    cols = ['Cabin', 'Name', 'Ticket', 'PassengerId']
    for col in cols:
        df = df.drop(col, axis=1)
    return df


def map_sex(df):
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df


def map_embarked(df):
    df['Embarked'] = df['Embarked'].map(embarked_mapping)
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
df_questioned = preprocessing_pipeline(df_questioned)

# extract answers
df_src_survived = df_src.loc[:, 'Survived']
df_src = df_src.drop(['Survived'], axis=1)

# encode categorical features
ce_encoder = ce.OneHotEncoder(cols=['Pclass', 'Embarked'], drop_invariant=True, impute_missing=False)
ce_encoder.fit(df_src)
df_src = ce_encoder.transform(df_src)
df_questioned = ce_encoder.transform(df_questioned)

# split src data for testing later
X_train, X_test, y_train, y_test = train_test_split(df_src, df_src_survived, test_size=0.1, random_state=0)

# scale
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train = stdsc.transform(X_train)
X_test = stdsc.transform(X_test)


# skipped step:
# TODO: may be we need to choose meaningful fields using regularization or do PCA
# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)


# train models
def run_and_check(name, classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    print(name + ":\t%.3f" % classifier.score(X_test, y_test))


run_and_check('lr', LinearRegression(), X_train, y_train, X_test, y_test)
run_and_check('knbr', KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski'), X_train, y_train, X_test, y_test)

tree = DecisionTreeClassifier(max_depth=1, criterion='entropy')
run_and_check('tree', tree, X_train, y_train, X_test, y_test)

ada_tree = AdaBoostClassifier(base_estimator=tree, n_estimators=200, learning_rate=0.2)
run_and_check('ada_tree', ada_tree, X_train, y_train, X_test, y_test)
