import pandas as pd

from helpers import DataFrameImputer

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

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


def encode_categorical(df):
    # TODO: why should we treat Pclass as categorical?
    df = pd.get_dummies(df, columns=['Pclass', 'Embarked'])
    return df


def preprocessing_pipeline(df):
    df = drop_cols(df)
    df = DataFrameImputer().fit_transform(df)
    df = map_sex(df)
    df = map_embarked(df)
    df = encode_categorical(df)
    return df


df_train = preprocessing_pipeline(df_train)
df_test = preprocessing_pipeline(df_test)

print(df_test.tail())
