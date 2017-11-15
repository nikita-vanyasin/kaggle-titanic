
from sklearn import datasets
import pandas as pd


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print(df_train.head())
