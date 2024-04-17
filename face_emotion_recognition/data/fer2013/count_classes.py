import pandas as pd

dataframe = pd.read_csv('fer2013.csv')['emotion'].value_counts().sort_index()

print(dataframe)